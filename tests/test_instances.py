import json
import os
import signal

import llm_runtime.instances as instances


def test_register_and_active_instances(tmp_path, monkeypatch):
    monkeypatch.setattr(instances, "LOCK_DIR", tmp_path)

    instances.register_instance("model-a.gguf", 4.0)

    active = instances.active_instances()
    assert len(active) == 0  # exclu car c'est le process courant


def test_active_instances_excludes_self_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(instances, "LOCK_DIR", tmp_path)

    dead_pid = 99999999  # PID hors plage, certainement mort
    (tmp_path / f"{dead_pid}.json").write_text(json.dumps({
        "pid": dead_pid,
        "model": "other.gguf",
        "size_gb": 3.0,
        "started_at": 0,
    }))

    active = instances.active_instances()
    assert active == []
    assert not (tmp_path / f"{dead_pid}.json").exists()


def test_active_instances_keeps_live_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(instances, "LOCK_DIR", tmp_path)

    # Le process courant est vivant par définition
    own_pid = os.getpid()
    (tmp_path / f"{own_pid}.json").write_text(json.dumps({
        "pid": own_pid,
        "model": "self.gguf",
        "size_gb": 2.0,
        "started_at": 0,
    }))

    active = instances.active_instances(exclude_pid=-1)
    assert len(active) == 1
    assert active[0]["model"] == "self.gguf"


def test_check_memory_pressure_within_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(instances, "LOCK_DIR", tmp_path)

    warning = instances.check_memory_pressure(new_size_gb=4.0, available_gb=12.0)

    assert warning is None


def test_check_memory_pressure_exceeded(tmp_path, monkeypatch):
    monkeypatch.setattr(instances, "LOCK_DIR", tmp_path)

    parent_pid = os.getppid()  # vivant, et différent du process de test
    (tmp_path / f"{parent_pid}.json").write_text(json.dumps({
        "pid": parent_pid,
        "model": "other.gguf",
        "size_gb": 8.0,
        "started_at": 0,
    }))

    warning = instances.check_memory_pressure(new_size_gb=4.0, available_gb=12.0)

    assert warning is not None
    assert "other.gguf" in warning


def test_terminate_other_instances_sends_sigterm(tmp_path, monkeypatch):
    monkeypatch.setattr(instances, "LOCK_DIR", tmp_path)

    pid = os.getppid()
    (tmp_path / f"{pid}.json").write_text(json.dumps({
        "pid": pid,
        "model": "other.gguf",
        "size_gb": 4.0,
        "started_at": 0,
    }))

    sent = []

    def fake_kill(target_pid, sig):
        if sig == 0:
            return
        sent.append((target_pid, sig))

    monkeypatch.setattr(instances.os, "kill", fake_kill)

    stopped = instances.terminate_other_instances(timeout_s=0)

    assert stopped[0]["pid"] == pid
    assert (pid, signal.SIGTERM) in sent


def test_terminate_other_instances_escalates_to_sigkill(tmp_path, monkeypatch):
    monkeypatch.setattr(instances, "LOCK_DIR", tmp_path)

    pid = os.getppid()
    (tmp_path / f"{pid}.json").write_text(json.dumps({
        "pid": pid,
        "model": "stubborn.gguf",
        "size_gb": 4.0,
        "started_at": 0,
    }))

    sent = []

    def fake_kill(target_pid, sig):
        if sig == 0:
            return
        sent.append((target_pid, sig))

    monkeypatch.setattr(instances.os, "kill", fake_kill)

    instances.terminate_other_instances(timeout_s=0)

    assert (pid, signal.SIGTERM) in sent
    assert (pid, signal.SIGKILL) in sent
