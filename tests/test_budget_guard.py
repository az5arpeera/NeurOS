"""The AWS budget guard (infra/aws): its Lambda logic with fake AWS clients, and its template.

No AWS access and no boto3 needed: ``budget_guard._client`` is replaced by fakes that record
every call, so these tests show exactly what the guard would do to an account.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from .conftest import REPO_ROOT

INFRA = REPO_ROOT / "infra" / "aws"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load("budget_guard", INFRA / "budget_guard.py")
build = _load("build_template", INFRA / "build_template.py")


# ---------------------------------------------------------------------- fakes
class _ParameterNotFoundError(Exception):
    pass


class FakeSSM:
    class exceptions:  # noqa: N801 - mirrors boto3's client.exceptions namespace
        ParameterNotFound = _ParameterNotFoundError

    def __init__(self, value: str | None = None) -> None:
        self.value = value
        self.puts: list[dict[str, Any]] = []

    def get_parameter(self, Name: str) -> dict[str, Any]:
        if self.value is None:
            raise _ParameterNotFoundError(Name)
        return {"Parameter": {"Name": Name, "Value": self.value}}

    def put_parameter(self, **kwargs: Any) -> None:
        self.puts.append(kwargs)
        self.value = kwargs["Value"]


class FakeEC2:
    """One region's instances: ``{id: state}``; ``unstoppable`` ids fail like one-time Spot."""

    def __init__(self, instances: dict[str, str], unstoppable: set[str] | None = None) -> None:
        self.instances = dict(instances)
        self.unstoppable = unstoppable or set()
        self.stop_calls: list[list[str]] = []
        self.fail_listing = False

    def get_paginator(self, name: str) -> Any:
        assert name == "describe_instances"
        owner = self

        class Paginator:
            def paginate(self, Filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
                if owner.fail_listing:
                    raise RuntimeError("UnauthorizedOperation")
                wanted = set(Filters[0]["Values"])
                ids = [i for i, state in owner.instances.items() if state in wanted]
                # Two pages, to exercise pagination.
                half = len(ids) // 2
                return [
                    {"Reservations": [{"Instances": [{"InstanceId": i} for i in ids[:half]]}]},
                    {"Reservations": [{"Instances": [{"InstanceId": i} for i in ids[half:]]}]},
                ]

        return Paginator()

    def stop_instances(self, InstanceIds: list[str]) -> None:
        self.stop_calls.append(list(InstanceIds))
        bad = [i for i in InstanceIds if i in self.unstoppable]
        if bad:
            raise RuntimeError(f"UnsupportedOperation: {bad}")
        for instance in InstanceIds:
            self.instances[instance] = "stopping"

    def describe_regions(self, AllRegions: bool) -> dict[str, Any]:
        assert AllRegions is False  # only regions enabled for the account
        return {"Regions": [{"RegionName": r} for r in ("us-east-1", "us-west-2")]}


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake account: two regions with instances in several states, and SSM."""
    regions = {
        "us-east-1": FakeEC2({"i-gpu": "running", "i-new": "pending", "i-off": "stopped"}),
        "us-west-2": FakeEC2({"i-forgot": "running"}),
    }
    ssm = FakeSSM()

    def client(service: str, region: str | None = None) -> Any:
        if service == "ssm":
            return ssm
        return regions[region or "us-east-1"]

    monkeypatch.setattr(guard, "_client", client)
    return {"regions": regions, "ssm": ssm}


SNS_ALERT = {
    "Records": [{"EventSource": "aws:sns", "Sns": {"Subject": "AWS Budgets: neuros-monthly-guard"}}]
}


def _start_event(instance: str, region: str = "us-east-1") -> dict[str, Any]:
    return {
        "detail-type": "EC2 Instance State-change Notification",
        "region": region,
        "detail": {"instance-id": instance, "state": "running"},
    }


# ---------------------------------------------------------------------- the Lambda's logic
def test_budget_alert_locks_the_month_and_stops_everything(account: dict[str, Any]) -> None:
    result = guard.handler(SNS_ALERT)
    month = guard._this_month()
    assert result["locked"] == month
    assert account["ssm"].value == month
    east, west = account["regions"]["us-east-1"], account["regions"]["us-west-2"]
    assert sorted(east.stop_calls[0]) == ["i-gpu", "i-new"]  # running + pending, one call
    assert west.stop_calls == [["i-forgot"]]  # every enabled region
    assert east.instances["i-off"] == "stopped"  # nothing else touched
    assert result["failed"] == {}


def test_dry_runs_change_nothing(account: dict[str, Any]) -> None:
    result = guard.handler({**SNS_ALERT, "dry_run": True})
    assert result["dry_run"] is True
    assert account["ssm"].puts == []
    assert all(not ec2.stop_calls for ec2 in account["regions"].values())
    # A plain test event ({} in the Lambda console) is always a dry run.
    report = guard.handler({})
    assert report["event"] == "test"
    assert report["locked"] is None
    assert all(not ec2.stop_calls for ec2 in account["regions"].values())


def test_instances_started_after_the_alert_are_stopped(account: dict[str, Any]) -> None:
    east = account["regions"]["us-east-1"]
    # Before any alert: starting a machine is allowed.
    assert guard.handler(_start_event("i-gpu"))["allowed"] is True
    assert east.stop_calls == []
    # Locked for an earlier month: allowed (budgets reset monthly).
    account["ssm"].value = "1999-01"
    assert guard.handler(_start_event("i-gpu"))["allowed"] is True
    # Locked for this month: stopped at once.
    account["ssm"].value = guard._this_month()
    result = guard.handler(_start_event("i-gpu"))
    assert result == {"event": "instance-start", "allowed": False, "stopped": "i-gpu"}
    assert east.stop_calls == [["i-gpu"]]


def test_unstoppable_instances_are_reported_and_fail_the_invocation(
    account: dict[str, Any],
) -> None:
    east = account["regions"]["us-east-1"]
    east.unstoppable = {"i-new"}  # e.g. a one-time Spot instance
    with pytest.raises(RuntimeError, match="i-new"):
        guard.handler(SNS_ALERT)
    # The batch failed, so each was retried alone: the stoppable one still stopped.
    assert east.instances["i-gpu"] == "stopping"
    assert account["regions"]["us-west-2"].instances["i-forgot"] == "stopping"
    assert account["ssm"].value == guard._this_month()  # the lock is set regardless


def test_a_region_that_cannot_be_listed_is_reported(account: dict[str, Any]) -> None:
    account["regions"]["us-west-2"].fail_listing = True
    with pytest.raises(RuntimeError, match="us-west-2"):
        guard.handler(SNS_ALERT)
    assert account["regions"]["us-east-1"].instances["i-gpu"] == "stopping"  # others continue


def test_the_guard_never_terminates_or_deletes() -> None:
    source = (INFRA / "budget_guard.py").read_text(encoding="utf-8")
    for forbidden in ("terminate_instances", "delete_volume", "delete_snapshot", "delete_"):
        assert forbidden not in source


# ---------------------------------------------------------------------- the template
class _CfnLoader(yaml.SafeLoader):
    """Reads CloudFormation's short-form tags (!Ref, !Sub, !GetAtt) as plain values."""


def _cfn_tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        return {suffix: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {suffix: loader.construct_sequence(node)}
    return {suffix: loader.construct_mapping(node)}  # type: ignore[arg-type]


_CfnLoader.add_multi_constructor("!", _cfn_tag)


@pytest.fixture(scope="module")
def template() -> dict[str, Any]:
    text = (INFRA / "budget-guard.yaml").read_text(encoding="utf-8")
    loaded = yaml.load(text, Loader=_CfnLoader)
    assert isinstance(loaded, dict)
    return loaded


def test_template_inlines_exactly_the_tested_code(template: dict[str, Any]) -> None:
    assert build.main(["--check"]) == 0
    inline = template["Resources"]["GuardFunction"]["Properties"]["Code"]["ZipFile"]
    source = (INFRA / "budget_guard.py").read_text(encoding="utf-8")
    assert inline.strip() == source.strip()


def test_template_settings(template: dict[str, Any]) -> None:
    resources = template["Resources"]
    budget = resources["GuardBudget"]["Properties"]["Budget"]
    # With credits counted (AWS's default), credit-funded spending would never trip the guard.
    assert budget["CostTypes"]["IncludeCredit"] is False
    assert budget["TimeUnit"] == "MONTHLY"
    function = resources["GuardFunction"]["Properties"]
    assert (function["Runtime"], function["Handler"]) == ("python3.12", "index.handler")
    statements = resources["GuardRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    actions = {a for s in statements for a in s["Action"]}
    assert "ec2:StopInstances" in actions
    assert not any("Terminate" in a or "Delete" in a for a in actions)  # stop only
    rule = resources["InstanceStartRule"]["Properties"]["EventPattern"]
    assert rule["detail"]["state"] == ["running"]
    assert template["Parameters"]["GuardThresholdPercent"]["Default"] == 90


def test_render_rejects_a_template_without_one_marker() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build.render("no marker here\n", "print('x')\n")


# ---------------------------------------------------------------------- the shell scripts
@pytest.mark.parametrize("script", sorted((REPO_ROOT / "scripts" / "aws").glob("*.sh")))
def test_shell_scripts_parse(script: Path) -> None:
    bash = shutil.which("bash")
    # From PowerShell, `bash` is WSL's launcher, which may have no Linux distribution installed.
    if (
        bash is None
        or subprocess.run([bash, "-c", "true"], capture_output=True, check=False).returncode
    ):
        pytest.skip("no working bash")
    # The script goes in on stdin, as bytes: WSL's bash can't open a Windows path, and text mode
    # would turn the script's LF line endings into CRLF on Windows.
    result = subprocess.run(
        [bash, "-n"], input=script.read_bytes(), capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
