# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import time
from importlib import import_module

import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform

platforms.current_platform = CpuPlatform()

vllm = import_module("vllm")
v1_engine = import_module("vllm.v1.engine")
engine_core = import_module("vllm.v1.engine.core")
output_processor_module = import_module("vllm.v1.engine.output_processor")
stats_module = import_module("vllm.v1.metrics.stats")
request_module = import_module("vllm.v1.request")

PoolingParams = vllm.PoolingParams
SamplingParams = vllm.SamplingParams
EngineCoreEventType = v1_engine.EngineCoreEventType
EngineCoreRequest = v1_engine.EngineCoreRequest
FinishReason = v1_engine.FinishReason
EngineCoreProc = engine_core.EngineCoreProc
OutputProcessor = output_processor_module.OutputProcessor
IterationStats = stats_module.IterationStats
Request = request_module.Request
RequestStatus = request_module.RequestStatus


def test_send_abort_outputs_preserves_request_metadata():
    engine = object.__new__(EngineCoreProc)
    engine.output_queue = queue.Queue()

    text_request = Request(
        request_id="text-request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        client_index=7,
        trace_headers={"traceparent": "text-trace"},
    )
    text_request.status = RequestStatus.FINISHED_ABORTED
    text_request.num_cached_tokens = 11
    text_request.num_external_computed_tokens = 4
    text_request.num_nans_in_logits = 2
    text_request.record_event(EngineCoreEventType.QUEUED, timestamp=1.0)

    pooling_request = Request(
        request_id="pool-request",
        prompt_token_ids=[4, 5, 6],
        sampling_params=None,
        pooling_params=PoolingParams(task="embed"),
        client_index=7,
        trace_headers={"traceparent": "pool-trace"},
    )
    pooling_request.status = RequestStatus.FINISHED_ABORTED
    pooling_request.num_cached_tokens = 6
    pooling_request.record_event(EngineCoreEventType.SCHEDULED, timestamp=2.0)

    engine._send_abort_outputs([text_request, pooling_request])

    client_index, engine_core_outputs = engine.output_queue.get_nowait()

    assert client_index == 7
    assert engine_core_outputs.finished_requests == {
        text_request.request_id,
        pooling_request.request_id,
    }

    outputs_by_id = {
        output.request_id: output for output in engine_core_outputs.outputs
    }

    text_output = outputs_by_id[text_request.request_id]
    assert text_output.finish_reason == FinishReason.ABORT
    assert text_output.trace_headers == text_request.trace_headers
    assert text_output.num_cached_tokens == text_request.num_cached_tokens
    assert (
        text_output.num_external_computed_tokens
        == text_request.num_external_computed_tokens
    )
    assert text_output.num_nans_in_logits == text_request.num_nans_in_logits
    assert text_output.pooling_output is None
    assert text_output.events is not None
    assert text_output.events[0].type == EngineCoreEventType.QUEUED

    pooling_output = outputs_by_id[pooling_request.request_id]
    assert pooling_output.finish_reason == FinishReason.ABORT
    assert pooling_output.trace_headers == pooling_request.trace_headers
    assert pooling_output.num_cached_tokens == pooling_request.num_cached_tokens
    assert pooling_output.pooling_output is not None
    assert pooling_output.pooling_output.device.type == "cpu"
    assert pooling_output.pooling_output.numel() == 0
    assert pooling_output.events is not None
    assert pooling_output.events[0].type == EngineCoreEventType.SCHEDULED


def test_abort_output_propagates_to_request_output_stats_and_tracing(monkeypatch):
    output_processor = OutputProcessor(
        tokenizer=None,
        log_stats=True,
        tracing_enabled=True,
    )

    engine_request = EngineCoreRequest(
        request_id="pool-request",
        external_req_id="pool-request-ext",
        prompt_token_ids=list(range(10)),
        mm_features=None,
        sampling_params=None,
        pooling_params=PoolingParams(task="embed"),
        arrival_time=time.time() - 1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        trace_headers={"traceparent": "pool-trace"},
    )
    output_processor.add_request(engine_request, prompt=None)

    tracing_calls = []

    def trace_spy(engine_core_output, req_state, iteration_stats):
        tracing_calls.append(
            {
                "trace_headers": engine_core_output.trace_headers,
                "num_cached_tokens": req_state.num_cached_tokens,
                "finished_requests": len(iteration_stats.finished_requests),
            }
        )

    monkeypatch.setattr(output_processor, "do_tracing", trace_spy)

    request = Request(
        request_id=engine_request.request_id,
        prompt_token_ids=engine_request.prompt_token_ids,
        sampling_params=None,
        pooling_params=engine_request.pooling_params,
        client_index=0,
        arrival_time=engine_request.arrival_time,
        trace_headers=engine_request.trace_headers,
    )
    request.status = RequestStatus.FINISHED_ABORTED
    request.num_cached_tokens = 6
    request.num_external_computed_tokens = 2
    request.record_event(EngineCoreEventType.SCHEDULED, timestamp=1.0)

    abort_output = EngineCoreProc._make_abort_output(request)
    iteration_stats = IterationStats()

    processed_outputs = output_processor.process_outputs(
        [abort_output],
        engine_core_timestamp=time.monotonic(),
        iteration_stats=iteration_stats,
    )

    assert processed_outputs.reqs_to_abort == []
    assert len(processed_outputs.request_outputs) == 1

    request_output = processed_outputs.request_outputs[0]
    assert request_output.request_id == engine_request.external_req_id
    assert request_output.finished
    assert request_output.num_cached_tokens == 6

    assert iteration_stats.prompt_token_stats.cached_tokens == 6
    assert iteration_stats.prompt_token_stats.external_kv_transfer == 2
    assert len(iteration_stats.finished_requests) == 1
    assert iteration_stats.finished_requests[0].finish_reason == FinishReason.ABORT
    assert iteration_stats.finished_requests[0].num_cached_tokens == 6

    assert tracing_calls == [
        {
            "trace_headers": {"traceparent": "pool-trace"},
            "num_cached_tokens": 6,
            "finished_requests": 1,
        }
    ]


def test_generate_abort_output_propagates_to_request_output_stats_and_tracing(
    monkeypatch,
):
    output_processor = OutputProcessor(
        tokenizer=None,
        log_stats=True,
        tracing_enabled=True,
    )

    engine_request = EngineCoreRequest(
        request_id="generate-request",
        external_req_id="generate-request-ext",
        prompt_token_ids=list(range(12)),
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=time.time() - 1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        trace_headers={"traceparent": "generate-trace"},
    )
    output_processor.add_request(engine_request, prompt=None)

    tracing_calls = []

    def trace_spy(engine_core_output, req_state, iteration_stats):
        tracing_calls.append(
            {
                "trace_headers": engine_core_output.trace_headers,
                "num_cached_tokens": req_state.num_cached_tokens,
                "finished_requests": len(iteration_stats.finished_requests),
            }
        )

    monkeypatch.setattr(output_processor, "do_tracing", trace_spy)

    request = Request(
        request_id=engine_request.request_id,
        prompt_token_ids=engine_request.prompt_token_ids,
        sampling_params=engine_request.sampling_params,
        pooling_params=None,
        client_index=0,
        arrival_time=engine_request.arrival_time,
        trace_headers=engine_request.trace_headers,
    )
    request.status = RequestStatus.FINISHED_ABORTED
    request.num_cached_tokens = 9
    request.num_external_computed_tokens = 4
    request.record_event(EngineCoreEventType.SCHEDULED, timestamp=1.0)

    abort_output = EngineCoreProc._make_abort_output(request)
    iteration_stats = IterationStats()

    processed_outputs = output_processor.process_outputs(
        [abort_output],
        engine_core_timestamp=time.monotonic(),
        iteration_stats=iteration_stats,
    )

    assert processed_outputs.reqs_to_abort == []
    assert len(processed_outputs.request_outputs) == 1

    request_output = processed_outputs.request_outputs[0]
    assert request_output.request_id == engine_request.external_req_id
    assert request_output.finished
    assert request_output.num_cached_tokens == 9
    assert len(request_output.outputs) == 1
    assert request_output.outputs[0].finish_reason == "abort"
    assert request_output.outputs[0].token_ids == []

    assert iteration_stats.prompt_token_stats.cached_tokens == 9
    assert iteration_stats.prompt_token_stats.external_kv_transfer == 4
    assert len(iteration_stats.finished_requests) == 1
    assert iteration_stats.finished_requests[0].finish_reason == FinishReason.ABORT
    assert iteration_stats.finished_requests[0].num_cached_tokens == 9

    assert tracing_calls == [
        {
            "trace_headers": {"traceparent": "generate-trace"},
            "num_cached_tokens": 9,
            "finished_requests": 1,
        }
    ]
