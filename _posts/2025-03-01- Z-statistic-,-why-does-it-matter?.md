---
layout: post
title: "Z Statistic: Why Does It Matter?"
date: 2025-03-01
categories: [statistics, data-science, machine-learning, education]
---

Before we talk about any statistics, let's talk about standardization and why it's important. I will do my best to not use math!

## Why Standardization is Important

Imagine if we didn't have any agreed standards to measure distance.

Say if someone asks you how far is your home, you would probably say it's 300 bananas away, or if you asked me, I would say 400 blue whales away.

Or imagine a bread recipe that calls for 5 handfuls of flour and 2 mugs of milk. But what if my hands/mugs are bigger than yours? I bet you wouldn't want to try my bread.

I know you might think these are silly examples because measuring distance in miles or kilometers, or measuring ingredients in cups, are standards we've all agreed to and take for granted. When you say your home is 2 miles away and I say my home is 5 miles away, we know that my home is farther away. If a recipe calls for 5 cups of flour and 2 cups of milk, we know the bread we bake will be similar.

**Standardization makes our life easy because we can compare data and make sense of it.**

Since you intuitively understand that standardization makes life easy, you already know why the Z statistic matters! Yes!

## What is the Z Statistic Measuring?

Now you may ask what Z is measuring. Can I say my house is 300 Zs away? Nope - it's not measuring distances.

Z measures Standard Deviations (SD). I assume you already know what a SD is; if not, I would recommend reviewing it before proceeding.

When we say Z = 2, it means 2 SDs. You should be wondering, "2 SDs from what point?" That point is the mean of the dataset.

So a Z of 2 implies that a data point is 2 SDs away from the mean. Let this sink in for a moment.

## Z-Scores in Action

Let's use an example. Let's say School A has 10 students, the mean weight is 110 pounds, and the SD is 10 pounds. Say you weigh 90 pounds.

How many SDs are you away from the mean?
```
Z = (Your weight - Mean) / Standard Deviation
Z = (90 - 110) / 10 = -2
```

So you are 2 SDs below the mean (110 pounds), giving you a Z score of -2.

Take another example: your friend in the **SAME school** weighs 120 pounds. Their Z score will be:
```
Z = (120 - 110) / 10 = +1
```

So just by looking at the Z scores, you can say that you weigh less than your friend because you are 2 SDs below the mean and they are 1 SD above the mean.

## Why Go Through All This Fuss?

Say you have another friend in a completely **different school (School B)**. There are 5 students in that class. The mean weight is 130 pounds and the SD is 5 pounds. Your friend weighs 120 pounds.

Their Z score would be:
```
Z = (120 - 130) / 5 = -2
```

Now, can you compare your Z score of -2 with your friend's (from School B) Z score of -2 and draw any insights?

One might think, "No, you cannot compare the Z scores as the mean & SD of the 2 schools are different." In statistics, we say the distributions of the 2 datasets are different.

Your Z score of -2 means you're 2 SDs below a mean of 110 with an SD of 10.
Your friend's Z score of -2 means they're 2 SDs below a mean of 130 with an SD of 5.

But here's the insight: While we can't directly compare your weight (90 pounds) to your friend's weight (120 pounds) using Z scores, we CAN compare your relative positions within your respective schools. Both of you are equally "light" relative to your school populations!

## The Magic of Z-Score Standardization

**What if I tell you the mean of the Z scores in both schools is 0 and SD = 1?**

You might be thinking: "Wait, what?!"

Let's see this in action with a table comparing both School A and School B:

**School A (Mean = 110 lbs, SD = 10 lbs)**

| Student | Weight (lbs) | Z-score |
|---------|--------------|---------|
| Alex    | 90           | -2.0    |
| Bailey  | 95           | -1.5    |
| Casey   | 100          | -1.0    |
| Dana    | 105          | -0.5    |
| Eli     | 110          | 0.0     |
| Fran    | 115          | +0.5    |
| Gabe    | 120          | +1.0    |
| Harper  | 125          | +1.5    |
| Iman    | 130          | +2.0    |
| Jordan  | 135          | +2.5    |
| **Mean**    | **110**          | **0.0**     |
| **SD**      | **10**           | **1.0**     |

**School B (Mean = 130 lbs, SD = 5 lbs)**

| Student | Weight (lbs) | Z-score |
|---------|--------------|---------|
| Kelly   | 120          | -2.0    |
| Logan   | 125          | -1.0    |
| Morgan  | 130          | 0.0     |
| Noah    | 135          | +1.0    |
| Olivia  | 140          | +2.0    |
| **Mean**    | **130**          | **0.0**     |
| **SD**      | **5**            | **1.0**     |

Here you can see how the original weight distributions differ between schools, but after converting to Z-scores, the distributions become directly comparable:

{% include charts/before-after-standardization.html %}

Notice something remarkable! Though the mean and SD of the weights in both schools are different, the mean and SD of Z-scores are exactly the same: mean = 0 and SD = 1.

{% include charts/z-score-chart.html %}

The Z score distributions are identical across these different datasets because we've standardized them. No matter what the original distribution looked like, Z-scores always create a standardized distribution with mean 0 and standard deviation 1, allowing us to compare relative positions across completely different datasets.

This is the beauty of standardization! It creates a level playing field to analyze data and make decisions.

## Why This Matters in Practice

Here's just one example of why standardization is critical in making decisions using data.

Say you are training a machine learning model and you feed it data about age and salary. Age will be in the range of 1-100, but salary might be in the range of $80,000 to $100 million (or more).

Many ML algorithms such as KNN, SVM, or gradient descent would treat larger numbers as more "important," leading to biased models where salary dominates simply because of its larger scale.

The solution is to standardize the data to create distributions with mean = 0 and SD = 1 for each feature.

For example:
- Age 50 might have Z = 0.2
- Salary $1M might have Z = 1.5

With standardized data, the model weighs both features proportionally. Now, a $1M income (Z = 1.5) and age = 50 (Z = 0.2) are compared fairly, leading to more reliable models.

## Other Benefits in Machine Learning

Z-scores play a crucial role in machine learning for several other important reasons:

### Feature Scaling
Many machine learning algorithms perform poorly when features have different scales. Z-score standardization ensures all features contribute appropriately to the model.

### Outlier Detection
Z-scores provide a standardized way to identify outliers. Data points with Z-scores beyond ±3 are often considered outliers in many applications.

### Improved Convergence
Gradient-based optimization algorithms converge faster when features are on similar scales, reducing training time and improving model stability.

### Better Interpretability
When features are standardized, the coefficients in linear models become more directly comparable, making it easier to interpret which features have the strongest influence.

### Transfer Learning
Z-scores help when applying a model trained on one dataset to a different dataset by providing a consistent scale.

In essence, Z-scores help machine learning algorithms "speak the same language" across different features and datasets, making models more accurate, efficient, and interpretable.

ABOVE ALL It helps in making decisions that makes sense.
---

*Would love to hear from you, write to me at statisticalbits.gmail.com*
