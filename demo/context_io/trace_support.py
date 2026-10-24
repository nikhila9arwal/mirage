"""Optional demo instrumentation; graph construction lives in demo.py."""
from collections import Counter
from contextlib import contextmanager
import csv
import importlib
import json
from pathlib import Path


def slot_label(slot, num_slots):
    """Keep A/B for two slots; larger runs use their numeric slot IDs."""
    return 'AB'[slot] if num_slots == 2 else str(slot)


def slot_track_id(slot):
    # Perfetto treats numeric tid=0 as pid. With pid=1 that would merge slots
    # 0 and 1, turning otherwise valid per-slot nesting into overlapping slices.
    return slot + 1


def worker_track_base(num_slots):
    # Slot tracks occupy 1..num_slots; workers must stay outside that range.
    return max(100, num_slots + 1)


@contextmanager
def compile_options(buffer, delay_task, delay_us, out):
    # Keep accepting the original demo's single task ID as well as a list.
    delay_tasks = [delay_task] if isinstance(delay_task, int) else list(delay_task)
    module = importlib.import_module('mirage.mpk.persistent_kernel')
    original = module.get_compile_command

    def command(*args, **kwargs):
        cmd = original(*args, **kwargs)
        # Keep the explicit compute_90a -> sm_90a target already in the command.
        cmd = [arg for arg in cmd if arg != '-arch=sm_90a']
        if buffer is not None:
            cmd[1:1] = [f'-I{Path(__file__).resolve().parent}', '-DMIRAGE_END_TRACE',
                        f'-DEND_TRACE_BUFFER_ADDRESS={buffer.data_ptr()}ULL',
                        f'-DEND_TRACE_CAPACITY={buffer.shape[0] - 1}ULL',
                        f'-DEND_TRACE_DELAY_NS={delay_us * 1000}ULL',
                        # NVCC treats unescaped commas as separate -D options.
                        '-DEND_TRACE_DELAY_TASKS=' + r'\,'.join(str(task) for task in delay_tasks)]
        (out / 'compile_command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        return cmd

    module.get_compile_command = command
    try:
        yield
    finally:
        module.get_compile_command = original


def export_trace(buffer, graph, num_contexts, delay_us, out):
    raw = buffer.cpu()
    count, overflow = int(raw[0, 0]), int(raw[0, 1])
    capacity = raw.shape[0] - 1
    if not 0 < count <= capacity or overflow:
        raise RuntimeError(f'Device trace overflow/empty trace: {count} / {capacity}')
    records = raw[1:count + 1].tolist()
    assert all(r[-1] == 1 for r in records), 'Unpublished trace record'
    with (out / 'trace.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_ns', 'kind', 'epoch', 'task_or_slot', 'data', 'worker',
                         'aux_context_or_count', 'sm', 'block', 'thread', 'reserved', 'published'])
        writer.writerows(records)
    kinds = lambda k: [r for r in records if r[1] == k]
    # Record layouts: compute=(epoch,TaskId,DataId,worker), copy=(epoch,slot,...,context).
    loaded = {(r[3], r[2]): r for r in kinds(38)}
    saved = {(r[3], r[2]): r for r in kinds(39)}
    assert len(kinds(38)) == len(kinds(39)) == len(loaded) == len(saved) == num_contexts
    assert sorted(r[6] for r in loaded.values()) == list(range(num_contexts))
    assert sorted(r[6] for r in saved.values()) == list(range(num_contexts))
    slots = graph['data_slot_ids']
    num_slots = len(graph['request_slots'])
    # Worker and slot tracks must have different IDs, even above 100 slots.
    worker_base = worker_track_base(num_slots)
    starts = {(r[2], r[4]): r for r in kinds(4)}
    ends = {(r[2], r[4]): r for r in kinds(5)}
    expected = Counter((epoch, d) for slot, epoch in loaded for d in graph['request_slots'][slot]['data_ids'])
    assert Counter((r[2], r[4]) for r in kinds(4)) == expected
    assert Counter((r[2], r[4]) for r in kinds(5)) == expected
    origin = min(r[0] for r in records)
    us = lambda t: (t - origin) / 1000
    timeline, events = [], []

    def track(tid, name):
        events.append(dict(ph='M', name='thread_name', pid=1, tid=tid, args=dict(name=name)))

    def span(tid, name, begin, end, **args):
        assert end >= begin
        events.append(dict(ph='X', name=name, pid=1, tid=tid, ts=us(begin), dur=(end-begin)/1000, args=args))

    events.append(dict(ph='M', name='process_name', pid=1, args=dict(name='NAS ContextIO')))
    runtime_end = max(r[0] for r in records)
    for slot in range(num_slots):
        track(slot_track_id(slot), f'Slot {slot_label(slot, num_slots)} — logical nesting')
        span(slot_track_id(slot), 'persistent invocation (device trace extent)', origin, runtime_end)
    for worker in sorted({r[5] for r in kinds(4)}):
        track(worker_base + worker, f'Worker {worker} — actual execution')

    for (slot, epoch), admission in loaded.items():
        done = saved[slot, epoch]
        assert admission[6] == done[6]
        if epoch > 1:
            assert saved[slot, epoch-1][0] <= admission[0]
        ids = graph['request_slots'][slot]['data_ids']
        begin = min(starts[epoch, d][0] for d in ids)
        end = max(ends[epoch, d][0] for d in ids)
        assert admission[0] <= begin <= end <= done[0]
        timeline.append(dict(context=admission[6], slot=slot_label(slot, num_slots), epoch=epoch,
                             loaded_us=us(admission[0]), start_us=us(begin), done_us=us(end), saved_us=us(done[0])))
        span(slot_track_id(slot), f'context {admission[6]} / epoch {epoch}', admission[0], done[0], context=admission[6], epoch=epoch)
        for d in ids:
            a, b = starts[epoch, d], ends[epoch, d]
            assert a[5] == b[5]
            task_type = graph['resident_tasks'][graph['all_data'][d]['resident_task_id']]['task_type']
            operator = {122: 'embedding', 123: 'SiLU/multiply', 124: 'top-1',
                        125: 'W13', 126: 'W2', 154: 'RMSNorm', 155: 'linear',
                        165: 'checksum', 166: 'attention', 167: 'QKV',
                        168: 'W_O'}.get(task_type, f'task {task_type}')
            name = graph['all_data'][d]['outputs'][0]['base_ptr'] + ' ' + operator
            info = dict(context=admission[6], epoch=epoch, task=a[3], data=d, worker=a[5], sm=a[7])
            span(slot_track_id(slot), name, a[0], b[0], **info)
            span(worker_base + a[5], name, a[0], b[0], **info)
        for edge in graph['data_edges'] or []:
            src, dst = edge['src_data_id'], edge['dst_data_id']
            if slots[src] == slot:
                assert ends[epoch, src][0] <= starts[epoch, dst][0]

    overlaps, delays = [], []
    for begin in kinds(6):
        end, = [r for r in kinds(7) if r[2:5] == begin[2:5]]
        slot = slots[begin[4]]
        span(slot_track_id(slot), 'injected delay', begin[0], end[0], epoch=begin[2])
        span(worker_base + begin[5], 'injected delay', begin[0], end[0], epoch=begin[2])
        # Compare actual starts against each delayed slot's epoch, not queue IDs.
        newer = sorted((r for r in kinds(4) if slots[r[4]] != slot and
                        r[2] > begin[2] and begin[0] < r[0] < end[0]), key=lambda r: r[0])
        assert all(r[5] != begin[5] for r in newer)
        first = newer[0] if newer else None
        delays.append(dict(slot=slot, epoch=begin[2], worker=begin[5],
                           begin_us=us(begin[0]), end_us=us(end[0]),
                           duration_us=(end[0] - begin[0]) / 1000,
                           other_slots_in_newer_epochs=sorted({slots[r[4]] for r in newer}),
                           max_other_epoch=max((r[2] for r in newer), default=None),
                           first_newer_compute=None if first is None else dict(
                               slot=slots[first[4]], epoch=first[2], task=first[3],
                               data=first[4], worker=first[5], start_us=us(first[0]))))
        for start in newer:
            # Preserve the old report field specifically for slot 0 vs slot 1.
            if slot == 1 and slots[start[4]] == 0:
                overlaps.append(dict(A_epoch=start[2], B_epoch=begin[2], A_start_us=us(start[0]),
                                     B_delay_begin_us=us(begin[0]), B_delay_end_us=us(end[0])))
    for record in kinds(38) + kinds(39):
        events.append(dict(ph='i', s='t', name='input loaded' if record[1] == 38 else 'output saved',
                           pid=1, tid=slot_track_id(record[3]), ts=us(record[0]), args=dict(context=record[6], epoch=record[2])))
    # Complete slices nest by interval; equal starts put enclosing slices first.
    events.sort(key=lambda e: (e.get('ts', -1), -e.get('dur', 0)))
    (out / 'trace.perfetto.json').write_text(json.dumps(dict(traceEvents=events, displayTimeUnit='ms')) + '\n')
    timeline.sort(key=lambda row: row['context'])
    with (out / 'timeline.csv').open('w') as f:
        fields = ['context', 'slot', 'epoch', 'loaded_us', 'start_us', 'done_us', 'saved_us']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(timeline)
    print('\nActual execution timeline (microseconds from first device record):')
    print(' Context  Slot  Epoch      Start       Done      Saved')
    for row in timeline:
        print(f" {row['context']:7d}  {row['slot']:>4}  {row['epoch']:5d}  {row['start_us']:9.3f}  {row['done_us']:9.3f}  {row['saved_us']:9.3f}")
    for delay in delays:
        first = delay['first_newer_compute']
        if first is not None:
            print(f"Observed slot {slot_label(first['slot'], num_slots)} epoch {first['epoch']} "
                  f"executing during slot {slot_label(delay['slot'], num_slots)} epoch {delay['epoch']}'s delay.")
        else:
            print(f"No newer-epoch compute start observed during slot {delay['slot']} epoch {delay['epoch']}'s delay.")
    return dict(trace_records=count, trace_capacity=capacity, trace_overflow=overflow, admitted=num_contexts, saved=num_contexts,
                A_newer_epoch_during_B_delay=overlaps, delays=delays)
