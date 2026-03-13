import json
import os
import sys
from collections import defaultdict
from typing import Literal

import graphviz as gv
import matplotlib.colors as mcolors

base_event = 4294967294

# Exclude these because they make the text hard to see.
dark = {
    "blue",
    "blueviolet",
    "brown",
    "black",
    "midnightblue",
    "navy",
    "indigo",
    "mediumblue",
    "dimgrey",
    "dimgray",
}
svg_colors = [
    color for color in mcolors.CSS4_COLORS.keys()
    if color not in dark and "dark" not in color
]


def get_color_map(prefix: Literal["TASK", "EVENT"]) -> dict[int, tuple[str, str]]:
    filename = os.path.join(
        os.environ["MIRAGE_HOME"],
        "include/mirage/persistent_kernel/runtime_header.h",
    )
    with open(filename, "r") as f:
        lines = f.readlines()
    result = {}
    keep_processing = False
    for line in lines:
        if keep_processing and prefix in line:
            name = line.strip().split(" ")[0][len(prefix) + 1:]
            number = int(line.strip().split(" ")[2][:-1])
            result[number] = (svg_colors[len(result)], name)
        elif f"enum {prefix}".upper() in line.upper():
            keep_processing = True
        elif "}" in line:
            keep_processing = False
    return result


supported_data_types = {
    940: ("float16", 2),
    941: ("bfloat16", 2),
    950: ("float32", 4),
    965: ("int64", 8),
}


def check_supported_data_types() -> None:
    filename = os.path.join(os.environ["MIRAGE_HOME"], "include/mirage/type.h")
    with open(filename, "r") as f:
        lines = f.readlines()
    lines_set = {line.strip() for line in lines}
    for number, (name, _) in supported_data_types.items():
        expected = f"DT_{name.upper()} = {number},"
        assert expected in lines_set, f"{expected} is not found in {filename}"


def get_offset_in_number_of_elements(tensor_json: dict) -> int:
    assert tensor_json["data_type"] in supported_data_types
    return tensor_json["offset"] / supported_data_types[tensor_json["data_type"]][1]


def get_index_from_id(identifier: int) -> int:
    return identifier & 0xFFFFFFFF


def tensor_lines(tensors: list[dict], prefix: str) -> str:
    if not tensors:
        return f"{prefix}: None"
    lines = []
    for tensor in tensors:
        if tensor["data_type"] in supported_data_types:
            offset = get_offset_in_number_of_elements(tensor)
        else:
            offset = tensor.get("offset", 0)
        lines.append(f"{prefix}: {tensor['base_ptr']} + {offset}")
    return "\n".join(lines)


def metadata_lines(desc: dict) -> list[str]:
    fields = []
    for key in [
        "request_id",
        "expert_offset",
        "kv_idx",
        "merge_task_offset",
        "task_offset",
    ]:
        value = desc.get(key, -1)
        if value not in (-1, None):
            fields.append(f"{key}: {value}")
    return fields


def render_legacy_graph(task_graph: dict,
                        graph: gv.Digraph,
                        task_type_color_map,
                        event_type_color_map) -> None:
    for event_idx, event in enumerate(task_graph["all_events"]):
        description = (
            f"event_idx: {event_idx}\n"
            f"event_type: {event_type_color_map[event['event_type']][1]}\n"
            f"num_triggers: {event['num_triggers']}\n"
            f"first_task_id: {event['first_task_id']}\n"
            f"last_task_id: {event['last_task_id']}"
        )
        graph.attr(
            "node",
            fillcolor=event_type_color_map[event["event_type"]][0],
            style="filled",
            shape="box",
        )
        graph.node(f"event_{event_idx}", description)
    graph.node(f"event_{base_event}", "Base Event")

    for task_idx, task in enumerate(task_graph["all_tasks"]):
        description = "\n".join([
            f"task_idx: {task_idx}",
            tensor_lines(task.get("inputs") or [], "in"),
            tensor_lines(task.get("outputs") or [], "out"),
            f"task_type: {task_type_color_map[task['task_type']][1]}",
            f"variant_id: {task['variant_id']}",
        ] + metadata_lines(task))
        graph.attr(
            "node",
            fillcolor=task_type_color_map[task["task_type"]][0],
            style="filled",
            shape="rectangle",
        )
        graph.node(f"task_{task_idx}", description)

        dependent_event_idx = get_index_from_id(task["dependent_event"])
        trigger_event_idx = get_index_from_id(task["trigger_event"])
        graph.edge(f"event_{dependent_event_idx}", f"task_{task_idx}")
        graph.edge(f"task_{task_idx}", f"event_{trigger_event_idx}")


def render_resident_data_graph(task_graph: dict,
                               graph: gv.Digraph,
                               task_type_color_map,
                               event_type_color_map) -> None:
    resident_tasks = task_graph.get("resident_tasks", [])
    all_data = task_graph.get("all_data", [])
    data_edges = task_graph.get("data_edges", [])
    first_data_ids = set(task_graph.get("first_data_ids", []))
    control_events = task_graph.get("control_events") or task_graph.get("all_events", [])

    graph.attr("node", shape="box", style="filled", fillcolor="lightgrey")
    graph.node("control_start", "BEGIN_TASK_GRAPH")
    graph.node("control_end", "END_OF_TASK_GRAPH")

    control_event_types = {
        event.get("event_type"): idx for idx, event in enumerate(control_events)
    }
    for event_idx, event in enumerate(control_events):
        event_name = event_type_color_map.get(event["event_type"], ("white", f"EVENT_{event['event_type']}"))[1]
        description = (
            f"control_event_idx: {event_idx}\n"
            f"event_type: {event_name}\n"
            f"num_triggers: {event.get('num_triggers', 0)}"
        )
        graph.attr(
            "node",
            shape="box",
            style="filled",
            fillcolor=event_type_color_map.get(event["event_type"], ("white", ""))[0],
        )
        graph.node(f"control_event_{event_idx}", description)

    for resident_task_id, resident_task in enumerate(resident_tasks):
        color, task_name = task_type_color_map[resident_task["task_type"]]
        description = "\n".join([
            f"resident_task_id: {resident_task_id}",
            f"task_type: {task_name}",
            f"variant_id: {resident_task['variant_id']}",
            f"num_inputs: {resident_task['num_inputs']}",
            f"num_outputs: {resident_task['num_outputs']}",
            f"max_parallelism: {resident_task['max_parallelism']}",
            f"total_data_count: {resident_task['total_data_count']}",
        ])
        graph.attr("node", shape="rectangle", style="filled", fillcolor=color)
        graph.node(f"resident_{resident_task_id}", description)

    outgoing = defaultdict(int)
    for edge in data_edges:
        outgoing[int(edge["src_data_id"])] += 1

    for data_id, data_desc in enumerate(all_data):
        resident_task_id = int(data_desc["resident_task_id"])
        resident_task = resident_tasks[resident_task_id]
        color, task_name = task_type_color_map[resident_task["task_type"]]
        description = "\n".join([
            f"data_id: {data_id}",
            f"resident_task_id: {resident_task_id}",
            f"task_type: {task_name}",
            f"initial_predecessor_count: {data_desc['initial_predecessor_count']}",
            tensor_lines(data_desc.get("inputs") or [], "in"),
            tensor_lines(data_desc.get("outputs") or [], "out"),
        ] + metadata_lines(data_desc))
        graph.attr("node", shape="ellipse", style="filled", fillcolor=color)
        graph.node(f"data_{data_id}", description)
        graph.edge(f"resident_{resident_task_id}",
                   f"data_{data_id}",
                   style="dotted",
                   color="grey40")
        if data_id in first_data_ids:
            graph.edge("control_start", f"data_{data_id}")
        if outgoing[data_id] == 0:
            graph.edge(f"data_{data_id}", "control_end")

    for edge in data_edges:
        graph.edge(f"data_{edge['src_data_id']}", f"data_{edge['dst_data_id']}")

    begin_event_idx = control_event_types.get(903)
    end_event_idx = control_event_types.get(910)
    if begin_event_idx is not None:
        graph.edge(f"control_event_{begin_event_idx}", "control_start", style="dashed")
    if end_event_idx is not None:
        graph.edge("control_end", f"control_event_{end_event_idx}", style="dashed")


def display_task_graph(task_graph_json_filename: str, use_xdot: bool) -> None:
    check_supported_data_types()
    task_type_color_map = get_color_map("TASK")
    event_type_color_map = get_color_map("EVENT")
    with open(task_graph_json_filename, "r") as file:
        task_graph = json.load(file)

    graph = gv.Digraph()
    graph.attr(rankdir="LR")

    if task_graph.get("schema_version", 1) >= 2 and task_graph.get("resident_tasks"):
        render_resident_data_graph(
            task_graph, graph, task_type_color_map, event_type_color_map
        )
    else:
        render_legacy_graph(task_graph, graph, task_type_color_map, event_type_color_map)

    dot_filename = task_graph_json_filename.replace(".json", ".dot")
    graph.save(dot_filename)
    print(f"Graph's dot representation saved as {dot_filename}")
    if use_xdot:
        os.system(f"xdot {dot_filename}")
    else:
        print("Consider installing a dot file viewer such as xdot to view/search the graph")
    print("Done rendering")


if __name__ == "__main__":
    display_task_graph(sys.argv[1], use_xdot=True)
