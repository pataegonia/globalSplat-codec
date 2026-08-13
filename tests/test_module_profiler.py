import time

from globalsplat.misc.module_profiler import profile_call, summarize_module_profiles


def test_cpu_profile_call_and_summary():
    output, sample = profile_call(lambda: (time.sleep(0.001), 7)[1], device="cpu")

    assert output == 7
    assert sample["seconds"] >= 0.001
    assert sample["cuda_peak_allocated_bytes"] == 0
    summary = summarize_module_profiles(
        [{"module": sample}, {"module": sample}], warmup_steps=1
    )
    assert summary["scenes_total"] == 2
    assert summary["scenes_profiled"] == 1
    assert summary["modules"]["module"]["calls"] == 1
    assert summary["modules"]["module"]["mean_seconds"] == sample["seconds"]
