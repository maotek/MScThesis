## Notes on failed jobs

- Some jobs fail randomly because `wandb` tries to read from `stdin` while `tqdm` is also writing to the terminal stream, which can break in non-interactive runs.
- Solution: Disable the progress bar with `tqdm.tqdm(..., disable=True)`.
