## Installation

To install the [`vision-mech-interp-attn-head`](https://pypi.org/project/vision-mech-interp-attn-head/) package, run the following command:

```bash
pip install vision-mech-interp-attn-head
```
---

### Citation
Bahador N. Mechanistic interpretability of fine-tuned vision transformers on distorted images: Decoding attention head behavior for transparent and trustworthy AI. arXiv [csLG]. Published online 24 March 2025. http://arxiv.org/abs/2503.18762. 

**[ArXiv Publication](https://arxiv.org/pdf/2503.18762)**

---

### Usage Example

Here is an example of how to use this package:

#### Quick Links:
- **Python Data Generation Package**: [Download here](https://lnkd.in/gVB6EUUV)
- **Sample Dataset**: [Download here](https://lnkd.in/gzX5vfVq)
- **Pre-trained Model**: [Download here](https://lnkd.in/gBgqriqa)
- **PyTorch Fine-Tuning Code**: [View implementation](https://lnkd.in/gc4QEYNq)

```python
import torch
import os
import pandas as pd 
from torch.utils.data import DataLoader
from torchvision import transforms
from vision_mech_interp_attn_head import (
    ViTForRegression, SpectrogramDataset, evaluate_model,
    perform_ablation_analysis, perform_baseline_analysis, plot_heatmap, extract_distributions, compare_distributions
)

# Directory containing the saved model and spectrogram images
data_dir = "YOUR_DATA_DIR"  # Replace with the path to your data directory
output_dir = "YOUR_OUTPUT_DIR"  # Replace with the path to your desired output directory

# Ensure output directory exists
print("Creating output directory if it doesn't exist...")
os.makedirs(output_dir, exist_ok=True)

# Load the saved fine-tuned model
model_path = os.path.join(data_dir, "best_vit_regression.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the fine-tuned model
print("Loading the fine-tuned model...")
model = ViTForRegression().to(device)

# Adjust state dictionary keys to match the saved model
state_dict = torch.load(model_path, map_location=device)
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("vit.base_model.model."):
        new_key = key.replace("vit.base_model.model.", "vit.")
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# Load the adjusted state dictionary
model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to ignore missing keys
model.eval()
print("Model loaded and set to evaluation mode.")

# Load the CSV file with labels
csv_path = os.path.join(data_dir, "labels.csv")
print(f"Loading labels from CSV file: {csv_path}...")
df = pd.read_csv(csv_path)

# Extract labels and indices
labels = df[["Chirp_Start_Time", "Chirp_Start_Freq", "Chirp_End_Freq"]].values
indices = df.index.values
print(f"Loaded {len(labels)} labels and {len(indices)} indices.")

# Create the dataset and dataloader
dataset = SpectrogramDataset(data_dir, indices, labels, device=device)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Adjust batch_size as needed

# Perform baseline analysis
perform_baseline_analysis(model, dataloader, output_dir)

# Perform ablation analysis
perform_ablation_analysis(model, dataloader, output_dir)

# Plot heatmap
plot_heatmap(os.path.join(output_dir, "ablation_results.csv"), baseline_loss=2465.65, output_path=os.path.join(output_dir, "heatmap.png"))

# Extract distributions
extract_distributions(output_dir, output_dir)

# Compare distributions
compare_distributions(
    os.path.join(output_dir, "combined_chirp_start_times.csv"),
    os.path.join(output_dir, "baseline.csv"),
    os.path.join(output_dir, "layer_head_distribution_grid.png")
)
```
