import matplotlib.pyplot as plt
from matplotlib import font_manager

# Set the font properties
font_path = '/path/to/SimHei.ttf'  # Replace with the actual path
fontP = font_manager.FontProperties(fname=font_path)

# Example plot
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6], label='Example Label')
plt.title('标题', fontproperties=fontP)
plt.xlabel('X轴', fontproperties=fontP)
plt.ylabel('Y轴', fontproperties=fontP)
plt.legend(prop=fontP)
plt.show()