import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df1 = pd.read_csv("Plotting/learning_curve_wandb_data/wandb_export_2024-09-16T10_59_17.919-04_00.csv")
df2 = pd.read_csv("Plotting/learning_curve_wandb_data/wandb_export_2024-09-16T11_01_50.014-04_00.csv")

df3= pd.read_csv("Plotting/learning_curve_wandb_data/wandb_export_2024-09-16T11_08_44.717-04_00.csv")
df4= pd.read_csv("Plotting/learning_curve_wandb_data/wandb_export_2024-09-16T11_08_35.279-04_00.csv")


fig, ax1 = plt.subplots()

color = 'black'
ax1.set_xlabel('Global Step')
ax1.set_ylabel('Mean Reward', color=color)
ax1.plot(df1["global_step"], df1["effortless-blaze-23 - rollout/ep_rew_mean"], label="Mean Reward", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('TWR50', color=color)  # we already handled the x-label with ax1
ax2.plot(df4["global_step"], df4["effortless-blaze-23 - twr/twr_inner"]/1200*100, linestyle = 'dashed', label = "TWR50 Inner", color='cornflowerblue')
ax2.plot(df2["global_step"], df2["effortless-blaze-23 - twr/twr"]/1200*100, label = "TWR50", color = color)
ax2.plot(df3["global_step"], df3["effortless-blaze-23 - twr/twr_outer"]/1200*100, linestyle = 'dotted', label = "TWR50 Outer",  color = 'darkblue')
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.plot(df1["global_step"], df1["effortless-blaze-23 - rollout/ep_rew_mean"], label="Mean Reward")
plt.plot(df4["global_step"], df4["effortless-blaze-23 - twr/twr_inner"]/1200, label = "TWR50 Inner")
plt.plot(df2["global_step"], df2["effortless-blaze-23 - twr/twr"]/1200, label = "TWR50")
plt.plot(df3["global_step"], df3["effortless-blaze-23 - twr/twr_outer"]/1200, label = "TWR50 Outer")

plt.xlabel("Global Step")
plt.ylabel("Score")


plt.legend()


plt.show()
