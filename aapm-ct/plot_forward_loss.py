import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data into a pandas DataFrame
df = pd.read_csv("forward_log.csv")

# Plot for loss and val_loss
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['loss'], label='loss')
plt.plot(df.index, df['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('loss_plot.png')  # Save the plot as loss_plot.png


# Plot for rel_l2_error1 and val_rel_l2_error1
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['rel_l2_error1'], label='rel_l2_error1')
plt.plot(df.index, df['val_rel_l2_error1'], label='val_rel_l2_error1')
plt.xlabel('Epoch')
plt.ylabel('Relative L2 Error')
plt.title('Training and Validation Relative L2 Error')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('relative_error_plot.png')  # Save the plot as relative_error_plot.png
