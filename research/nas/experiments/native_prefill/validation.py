"""Check saved device evidence without trusting the scheduler's pass decisions."""
import torch

PAGE = 64


def check_run(data, history, lengths, consumed, predicted, visits, owners,
              captured_hidden, saved_hidden, slots):
    p, vocab, eos = data['chunk_size'], data['vocab_size'], data['eos_token_id']
    count = len(data['prompts'])
    history = history.reshape(count, PAGE)
    lengths = lengths.reshape(count)
    consumed = consumed.reshape(count, PAGE, p)
    predicted = predicted.reshape(count, PAGE)
    visits, owners = visits.reshape(count, PAGE), owners.reshape(count, PAGE)
    captured_hidden = captured_hidden.reshape(count, PAGE, -1)
    saved_hidden = saved_hidden.reshape(count, PAGE, -1)
    rows, passed = [], True
    for request, prompt in enumerate(data['prompts']):
        length, generated = prompt['length'], int(lengths[request])
        checks = dict(output_length=1 <= generated <= PAGE-length+1)
        if not checks['output_length']:
            rows.append(dict(request=request, passed=False, checks=checks, generated_length=generated))
            passed = False
            continue
        tokens = history[request, :generated]
        # A completed request consumes its whole prompt, then every prediction
        # except the final one. These positions are independent of captured epochs.
        starts = list(range(0, length, p)) + list(range(length, length+generated-1))
        accepted_starts = [length-p] + list(range(length, length+generated-1))
        used = torch.zeros(PAGE, dtype=torch.bool)
        used[starts] = True
        output_used = torch.arange(PAGE) < generated
        slot = int(owners[request, starts[0]])
        checks.update(
            prediction_ids=bool(((tokens >= 0) & (tokens < vocab)).all()),
            saved_final_token=bool(torch.equal(predicted[request, accepted_starts], tokens)),
            visits_exact=bool((visits[request, used] == 1).all() and (visits[request, ~used] == 0).all()),
            same_slot_for_request=0 <= slot < slots and bool((owners[request, used] == slot).all()),
            unused_owners=bool((owners[request, ~used] == -1).all()),
            unused_captures=bool((consumed[request, ~used] == -1).all() and
                                 (predicted[request, ~used] == -1).all()),
            unused_history=bool((history[request, ~output_used] == -1).all()),
            eos_not_consumed=not bool((tokens[:-1] == eos).any()),
            stopping=int(tokens[-1]) == eos or length+generated-1 == PAGE,
            saved_hidden_finite=bool(torch.isfinite(saved_hidden[request, output_used]).all()),
            saved_hidden_exact=torch.equal(saved_hidden[request, output_used],
                                           captured_hidden[request, accepted_starts]),
            unused_hidden=bool(torch.isnan(saved_hidden[request, ~output_used]).all()),
            every_pass_hidden_finite=bool(torch.isfinite(captured_hidden[request, used]).all()),
        )
        prompt_ok, feedback_ok = True, True
        for position in range(0, length, p):
            expected = torch.tensor(prompt['token_ids'][position:position+p], dtype=torch.int64)
            prompt_ok &= torch.equal(consumed[request, position], expected)
        for step, position in enumerate(range(length, length+generated-1)):
            expected = torch.full((p,), -1, dtype=torch.int64)
            expected[0] = tokens[step]
            feedback_ok &= torch.equal(consumed[request, position], expected)
        checks.update(all_prompt_tokens_consumed=prompt_ok, autoregressive_feedback=feedback_ok)
        ok = all(checks.values())
        passed &= ok
        rows.append(dict(request=request, passed=ok, checks=checks, slot=slot,
            prompt_length=length, generated_length=generated, consumed_rows=length+generated-1,
            prefill_passes=length//p, decode_passes=generated-1,
            graph_passes=len(starts), stopped_by_eos=int(tokens[-1]) == eos))
    return dict(passed=passed, requests=rows,
        completed_graph_passes=sum(row.get('graph_passes',0) for row in rows),
        consumed_rows=sum(row.get('consumed_rows',0) for row in rows),
        generated_tokens=int(lengths.sum()),
        replacement_observed=count <= slots or len({row.get('slot') for row in rows}) < count)


def compare_hf(history, lengths, reference):
    rows = []
    for request, length in enumerate(lengths.tolist()):
        actual = history[request, :length].tolist()
        expected = reference['generated_tokens'][request]
        prefix = 0
        for a, b in zip(actual, expected):
            if a != b:
                break
            prefix += 1
        exact = actual == expected
        rows.append(dict(request=request, exact=exact, nas_length=length, hf_length=len(expected),
            matching_prefix=prefix, first_difference=None if exact else prefix,
            matching_positions=sum(a == b for a,b in zip(actual, expected)),
            compared_positions=max(length,len(expected))))
    return dict(exact_sequences=sum(row['exact'] for row in rows), sequences=len(rows),
        matching_positions=sum(row['matching_positions'] for row in rows),
        compared_positions=sum(row['compared_positions'] for row in rows), requests=rows,
        interpretation='Independent greedy paths; after a first mismatch their later inputs and KV differ. '
                       'Lifecycle correctness and HF numerical agreement are reported separately.')
