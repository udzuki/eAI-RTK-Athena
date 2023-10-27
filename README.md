# Athena

Requirements-driven Repair of Deep Neural Networks

## Dependencies

- [eAI-Repair-Toolkit](https://github.com/jst-qaml/eAI-Repair-Toolkit) >= 1.0

## Usage

1. Find `labels.json`.
2. Give requirements to `repair_priority` and `prevent_degradation` on each label.
3. Run the commands below:

```shell-session
(.venv)$ repair localize --method=Athena
(.venv)$ repair optimize --method=Athena
```

## License

[BSD 3-Clause](LICENSE)

----
&copy; 2020 [Udzuki, Inc.](https://www.udzuki.co.jp)
