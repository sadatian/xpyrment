import pandas as pd
import pytest

from xpyrment.cli import calculate_required_power, main


def test_calculate_required_power():
    """Asserts that sample size math is correct and rejects illegal parameter boundaries."""
    # Standard 80% power, 5% alpha, MDE=5%, standard deviation=20%
    nc, nt = calculate_required_power(mde=5.0, std=20.0, power=0.80, alpha=0.05, ratio=1.0)
    assert nc == 252  # Exact mathematical size
    assert nt == 252

    # Ratio change
    nc2, nt2 = calculate_required_power(mde=5.0, std=20.0, power=0.80, alpha=0.05, ratio=2.0)
    assert nc2 == 189
    assert nt2 == 378

    # Edge exceptions
    with pytest.raises(ValueError, match="must be strictly positive"):
        calculate_required_power(mde=-1.0, std=10.0)

    with pytest.raises(ValueError, match="must lie strictly between 0 and 1"):
        calculate_required_power(mde=1.0, std=10.0, power=1.5)


def test_cli_power_command(capsys):
    """Verifies that the CLI 'power' command executes cleanly and formats output to standard stdout."""
    # Call main with simulated arguments
    main(["power", "--mde", "0.05", "--std", "1.0", "--power", "0.80", "--alpha", "0.05", "--ratio", "1.0"])
    
    captured = capsys.readouterr()
    assert "POWER & REQUIRED SAMPLE SIZE" in captured.out
    assert "Required Control Group Size     : 6,280" in captured.out
    assert "Total Required Sample Size      : 12,560" in captured.out


def test_cli_balance_command(tmp_path, capsys):
    """Verifies that the CLI 'balance' command loads CSV data, checks covariate balance, and logs balance summaries."""
    # Generate mock CSV data
    df = pd.DataFrame({
        "assignment": ["control", "control", "treatment", "treatment"],
        "age": [25.0, 35.0, 26.0, 34.0],
        "tenure": [1.0, 2.0, 1.2, 1.8]
    })
    csv_file = tmp_path / "mock_data.csv"
    df.to_csv(csv_file, index=False)

    main(["balance", "--csv", str(csv_file), "--group-col", "assignment", "--covariates", "age, tenure"])

    captured = capsys.readouterr()
    assert "COVARIATE BALANCE REPORT (SMD)" in captured.out
    assert "Dataset Path" in captured.out
    assert "age" in captured.out
    assert "tenure" in captured.out
    assert "All targeted covariates are well balanced" in captured.out


def test_cli_regress_command(tmp_path, capsys):
    """Verifies that the CLI 'regress' command loads CSV data, runs OLS, and prints analytical results."""
    # Generate mock CSV
    df = pd.DataFrame({
        "revenue": [10.0, 12.0, 11.0, 15.0, 10.5, 12.5],
        "treatment": [0, 1, 0, 1, 0, 1],
        "age": [25, 30, 25, 30, 28, 29]
    })
    csv_file = tmp_path / "regression_data.csv"
    df.to_csv(csv_file, index=False)

    main(["regress", "--csv", str(csv_file), "--y-col", "revenue", "--x-cols", "treatment, age"])

    captured = capsys.readouterr()
    assert "OLS REGRESSION ANALYSIS REPORT" in captured.out
    assert "Intercept" in captured.out
    assert "treatment" in captured.out
    assert "age" in captured.out
    assert "Residual Degrees of Freedom:" in captured.out


def test_cli_nonexistent_csv_handling(capsys):
    """Asserts that calling CLI commands on nonexistent CSV paths reports friendly failures and exits securely."""
    with pytest.raises(SystemExit) as exc_info:
        main(["balance", "--csv", "nonexistent_file.csv", "--group-col", "group", "--covariates", "age"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error: CSV file not found" in captured.err


def test_cli_power_error_handling(capsys):
    """Verifies that invalid parameter inputs to the power subcommand are handled with a descriptive error and exit 1."""
    with pytest.raises(SystemExit) as exc_info:
        main(["power", "--mde", "-1.0", "--std", "1.0"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Power Calculation Error" in captured.err


def test_cli_balance_column_missing(tmp_path, capsys):
    """Verifies that the balance subcommand exits with 1 when the group or covariate columns are missing."""
    df = pd.DataFrame({"assignment": ["control", "treatment"], "age": [20, 30]})
    csv_file = tmp_path / "mock.csv"
    df.to_csv(csv_file, index=False)

    # Missing group column
    with pytest.raises(SystemExit) as exc_info:
        main(["balance", "--csv", str(csv_file), "--group-col", "wrong_col", "--covariates", "age"])
    assert exc_info.value.code == 1
    assert "Error: Group column" in capsys.readouterr().err

    # Missing covariate column
    with pytest.raises(SystemExit) as exc_info:
        main(["balance", "--csv", str(csv_file), "--group-col", "assignment", "--covariates", "missing_cov"])
    assert exc_info.value.code == 1
    assert "Error: Covariate column" in capsys.readouterr().err


def test_cli_regress_column_missing(tmp_path, capsys):
    """Verifies that the regress subcommand exits with 1 when the target or predictor columns are missing."""
    df = pd.DataFrame({"y": [1.0, 2.0], "x": [10.0, 20.0]})
    csv_file = tmp_path / "mock.csv"
    df.to_csv(csv_file, index=False)

    # Missing y column
    with pytest.raises(SystemExit) as exc_info:
        main(["regress", "--csv", str(csv_file), "--y-col", "wrong_y", "--x-cols", "x"])
    assert exc_info.value.code == 1
    assert "Dependent variable" in capsys.readouterr().err

    # Missing x column
    with pytest.raises(SystemExit) as exc_info:
        main(["regress", "--csv", str(csv_file), "--y-col", "y", "--x-cols", "missing_x"])
    assert exc_info.value.code == 1
    assert "Predictor variable" in capsys.readouterr().err


def test_cli_app_command_mocked(monkeypatch, capsys):
    """Verifies handle_app parses arguments, starts the hub server, attempts opening a browser, and shuts down cleanly."""
    server_started = []
    server_stopped = []
    browser_opened = []

    class MockHubServer:
        def __init__(self, host, port, log_usage):
            self.host = host
            self.port = port
            self.log_usage = log_usage

        def start(self):
            server_started.append(True)

        def stop(self):
            server_stopped.append(True)

    monkeypatch.setattr("xpyrment.run.hub.XpyrmentHubServer", MockHubServer)
    monkeypatch.setattr("webbrowser.open", lambda url: browser_opened.append(url))

    # Mock time.sleep to raise KeyboardInterrupt to break the infinite loop
    def mock_sleep(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("time.sleep", mock_sleep)

    main(["app", "--port", "8000", "--host", "127.0.0.1", "--log-usage"])

    assert True in server_started
    assert True in server_stopped
    assert "http://127.0.0.1:8000" in browser_opened
    
    captured = capsys.readouterr()
    assert "XPYRMENT PRIMARY HUB" in captured.out
    assert "Starting on http://127.0.0.1:8000" in captured.out
    assert "Usage logging is ENABLED" in captured.out
    assert "Shutting down primary hub server" in captured.out
