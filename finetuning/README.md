# FocusFeed summarization experiment

Open [FocusFeed-LoRA.ipynb](FocusFeed-LoRA.ipynb) in Colab and select a GPU runtime. For a smoke test, upload both JSONL files from [smoke](smoke/) into the notebook working directory and run all cells. These fictional examples check that training and evaluation run. In the initial smoke test, the base and LoRA variants both passed the automated checks on the two held-out articles.

For a quality evaluation, use 200–500 reviewed training pairs, a separate validation set for choosing settings, and at least 50 held-out articles. Keep related coverage of the same event in a single split. The notebook reads training and held-out data. Use the validation set in place of the held-out file during development, then restore the final held-out file for the comparison. Use the same checkpoint, prompts, articles, and generation settings for both variants.

Check faithfulness and citation handling with the judge, and compare its labels with your own blind ratings on 20 outputs. Measure summary length and JSON validity directly. Keep extraction coverage separate because this experiment starts with article text already available.

Before Colab disconnects, mount Drive and copy the adapter and report with a new code cell.

```python
from google.colab import drive
from datetime import datetime, timezone
from pathlib import Path
import shutil

drive.mount("/content/drive")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
destination = Path("/content/drive/MyDrive/FocusFeed") / stamp
destination.mkdir(parents=True)
shutil.copytree("focusfeed_results/adapter", destination / "adapter")
for name in ("report.json", "run_manifest.json"):
    shutil.copy2(Path("focusfeed_results") / name, destination / name)
print(destination)
```

## XSum pilot

The [XSum notebook](FocusFeed-XSum-Pilot.ipynb) prepares 30 training examples and 10 held-out BBC articles by default. It compares Qwen3-0.6B with its LoRA adapter enabled and disabled on the same test articles. Run the notebook in a GPU runtime to reproduce the experiment. The notebook downloads article evidence at runtime.

Across the 10 held-out articles, valid JSON increased from 1/10 to 10/10. Valid JSON within the 60-word limit increased from 1/10 to 8/10. The notebook's automated faithfulness and citation checks assign a failure to invalid JSON, so those scores combine formatting and content quality.

A separate blind AI review scored the readable outputs against the supplied evidence. Both variants passed 2/10 when scoring the headline, summary, and key points together, and 4/10 when scoring the summary alone. A more permissive assessment gave credit for capturing the main story on 10/10 base outputs and 8/10 LoRA outputs. These ratings came from AI review. The report contains zero human ratings. BBC URLs were derived from dataset IDs and were unverified.

The [pilot results](results/xsum-pilot/) include the original automated report, the variant key, and the two detailed AI score files. The notebook's original CSV field names are retained for compatibility. The training count of 30 is the notebook default. The run manifest was unavailable to confirm it.
