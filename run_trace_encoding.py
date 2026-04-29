import sys
from typing import Dict, Tuple, List
import numpy as np

import click
import pickle
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

import importlib

from data.trainticket.download import simple_name

# produces one row per trace; creates a fixed-sized matrix
# where each trace is flattened into a single feature vector

# Each service gets a slot (indexed by SERVICE2IDX) and the features
# (latency, http_status) are placed at that service's position. Services not
# present in a trace get -1.

# Used by trace-level classifiers (random forest, MLP) that answer
# "is this entire trace anomalouos?"

# Trace encoding summarizes a whole trace for classification

"""
Encode train-ticket pickle data into trace-level data and label:
{
'data': array in shape (n_traces, n_features * n_microservices),
'labels': array in shape (n_traces,)
'masks': array in shape (n_traces, n_features * n_microservices),
'trace_ids': array in shape (n_traces)
}
"""


def encoding_data(source_data: List, drop_service=(), drop_fault_type=(),
                  involved_services=None, service2idx=None, enable_all_features=False):
    def pair2index(s_t):
        return service2idx.get(simple_name(s_t[1]))

    if enable_all_features:
        _data = np.ones((len(source_data), len(involved_services), 9), dtype=np.float32) * -1
    else:
        _data = np.ones((len(source_data), len(involved_services), 2), dtype=np.float32) * -1

    _labels = np.zeros((len(source_data),), dtype=bool)
    _trace_ids = [""] * len(source_data)
    _service_mask = np.zeros((len(source_data), len(involved_services)), dtype=bool)
    _root_causes = np.zeros((len(source_data), len(involved_services)), dtype=bool)
    for trace_idx, trace in enumerate(source_data):
        if 'fault_type' in trace and trace['fault_type'] in drop_fault_type:
            continue
        if 'root_cause' in trace and any(_ in drop_service for _ in trace['root_cause']):
            continue
        indices = np.asarray([idx for idx, (source, target) in enumerate(trace['s_t']) if source != target])
        if len(indices) <= 0:
            continue
        for key, item in trace.items():
            if isinstance(item, list) and key != 'root_cause' and key != 'fault_type':
                trace[key] = np.asarray(item)[indices]
        service_idx = np.asarray(list(map(pair2index, (trace['s_t']))))
        _service_mask[trace_idx, service_idx] = True
        # assert all(np.diff(trace['endtime']) <= 0), f'end time is not sorted: {trace["endtime"]}'

        if enable_all_features:
            _data[trace_idx, service_idx, 0] = np.asarray(trace['latency']) / 1e6
            _data[trace_idx, service_idx, 1] = np.asarray(trace['cpu_use']) / 100
            _data[trace_idx, service_idx, 2] = np.asarray([round(_, 2) for _ in trace['mem_use_percent']])
            _data[trace_idx, service_idx, 3] = np.asarray(trace['mem_use_amount']) / 1e9  # 1000M
            _data[trace_idx, service_idx, 4] = np.asarray(trace['file_write_rate']) / 1e8
            _data[trace_idx, service_idx, 5] = np.asarray(trace['file_read_rate']) / 1e8
            _data[trace_idx, service_idx, 6] = np.asarray(trace['net_send_rate']) / 1e8
            _data[trace_idx, service_idx, 7] = np.asarray(trace['net_receive_rate']) / 1e8
            _data[trace_idx, service_idx, 8] = list(map(lambda x: x // 100 if x != 0 else 9, (trace['http_status'])))
        else:
            _data[trace_idx, service_idx, 0] = np.asarray(trace['latency'])
            _data[trace_idx, service_idx, 1] = list(map(lambda x: int(x) // 100 if x != 0 else 9, (trace['http_status'])))

        _labels[trace_idx] = trace['label']
        _trace_ids[trace_idx] = trace['trace_id']
        _trace_root_causes = trace['root_cause'] if 'root_cause' in trace else []
        for _root_cause in _trace_root_causes:
            _root_causes[trace_idx, service2idx[_root_cause]] = True
    _mask = np.tile(_service_mask[:, :, np.newaxis], (1, 1, 9))
    return _data, _labels, _mask, _trace_ids, _root_causes


@click.command('trace-encoding')
@click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
@click.option('-o', '--output', 'output_file', default='', type=str)
@click.option('--drop-service', default=0)
@click.option('--drop-fault-type', default=0)
@click.option('--dataset', default='tt', type=click.Choice(['tt', 'ob']),
              help='Dataset config: tt=Train-Ticket (default), ob=Online-Boutique')
def main(*args, **kwargs):
    train_ticket_trace_encoding(*args, **kwargs)


def train_ticket_trace_encoding(input_file: str, output_file: str, drop_service, drop_fault_type, dataset='tt'):
    cfg = importlib.import_module('trainticket_config' if dataset == 'tt' else 'onlineboutique_config')
    INVOLVED_SERVICES = cfg.INVOLVED_SERVICES
    FAULT_TYPES = cfg.FAULT_TYPES
    SERVICE2IDX = cfg.SERVICE2IDX
    ENABLE_ALL_FEATURES = cfg.ENABLE_ALL_FEATURES

    drop_service = list(INVOLVED_SERVICES)[:drop_service]
    drop_fault_type = list(FAULT_TYPES)[:drop_fault_type]
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)
    with open(str(input_file.resolve()), 'rb') as f:
        input_data = pickle.load(f)
    data, labels, masks, trace_ids, root_causes = encoding_data(
        input_data, drop_service, drop_fault_type,
        involved_services=INVOLVED_SERVICES,
        service2idx=SERVICE2IDX,
        enable_all_features=ENABLE_ALL_FEATURES,
    )
    if len(data) == 0:
        feature_width = data.shape[1] * data.shape[2]
        data = np.empty((0, feature_width), dtype=data.dtype)
        masks = np.empty((0, feature_width), dtype=masks.dtype)
    if len(data) == 0:
        np.savez(
            output_file,
            data=data,
            labels=labels,
            masks=masks,
            trace_ids=trace_ids,
            root_causes=root_causes,
        )
    else:
        np.savez(
            output_file,
            data=data.reshape((len(data), -1)),
            labels=labels,
            masks=masks.reshape((len(data), -1)),
            trace_ids=trace_ids,
            root_causes=root_causes,
        )


if __name__ == '__main__':
    main()
