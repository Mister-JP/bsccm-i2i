from pathlib import Path

from typer.testing import CliRunner

from bsccm_i2i.cli.main import app


def test_root_help_lists_expected_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "split" in result.stdout
    assert "train" in result.stdout
    assert "eval" in result.stdout
    assert "report" in result.stdout


def test_subcommand_help_exits_zero() -> None:
    runner = CliRunner()

    for command in ("split", "train", "eval", "report"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0


def test_eval_and_report_print_deterministic_called_message(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    expected = {
        "eval": "CALLED eval",
        "report": "CALLED report",
    }

    for command, line in expected.items():
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        assert line in result.stdout
