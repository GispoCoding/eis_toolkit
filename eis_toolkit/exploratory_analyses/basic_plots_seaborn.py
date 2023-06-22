import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from seaborn import histplot as sns_histogram

penguins = sns.load_dataset("penguins")
print(penguins)

sns.histplot(data=penguins, x="flipper_length_mm", hue="species", kde=True, palette="bright")
plt.show()


# sns.set_style("whitegrid")
# data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
# sns.boxplot(data=data)
# sns.despine(left=True)
# plt.show()
