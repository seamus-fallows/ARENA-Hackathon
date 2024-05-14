#%%
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib to use a font that supports Chinese characters
rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # Noto Sans CJK JP or Noto Sans CJK SC
rcParams['axes.unicode_minus'] = False  # Ensure that the minus sign is shown correctly

# Sample data
x = [0, 1, 2, 3, 4, 5]
y1 = [0, 1, 4, 9, 16, 25]
y2 = [0, 1, 8, 27, 64, 125]

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(x, y1, label='平方 (square)')
plt.plot(x, y2, label='立方 (cube)')

# Add title and labels
plt.title('示例图 (Sample Plot)')
plt.xlabel('x 轴 (x-axis)')
plt.ylabel('y 轴 (y-axis)')

# Add a legend
plt.legend()

# Show the plot
plt.show()



# %%
