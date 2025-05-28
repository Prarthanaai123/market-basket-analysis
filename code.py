import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# STEP 1: Load your one-hot encoded dataset
#df = pd.read_csv("basket_analysis.csv")
df = pd.read_csv("C:/Users/prart/OneDrive/Desktop/python project/basket_analysis.csv")


# If there's an unnamed index column, drop it
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Ensure data is boolean (TRUE/FALSE → True/False)
df = df.astype(bool)

# STEP 2: Apply Apriori Algorithm
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

# STEP 3: Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Print Rules
print("Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# STEP 4: Visualize Top 5 Rules by Confidence
rules_sorted = rules.sort_values(by='confidence', ascending=False).head(5)

plt.barh(range(len(rules_sorted)), rules_sorted['confidence'], color='skyblue')
plt.yticks(range(len(rules_sorted)), [f"{', '.join(list(a))} → {', '.join(list(c))}" for a, c in zip(rules_sorted['antecedents'], rules_sorted['consequents'])])
plt.xlabel("Confidence")
plt.title("Top 5 Association Rules")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
