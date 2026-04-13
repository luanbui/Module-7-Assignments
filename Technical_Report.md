# Food-101 Image Classification Using Deep Learning: A Comparative Study of CNN Architectures and Transfer Learning

**Authors:** Luan Bui, Ganguli, Sixpence

**Course:** CS/DS — Spring 2026

---

## 1. Introduction and Problem Definition

Classifying food from photographs is a practically valuable yet technically demanding computer vision task. Applications range from automated dietary tracking and calorie estimation to restaurant menu recognition and food delivery quality assurance. The core challenge is that food images exhibit fine-grained visual similarity — many dishes share colors, textures, and plating styles — while also varying widely in lighting, angle, and composition within a single category.

This project investigates multi-class food image classification using the Food-101 dataset. Given a photograph of a dish, the goal is to predict which of 101 food categories it belongs to. We approach this as a supervised classification problem and compare three neural network architectures of increasing sophistication: a shallow baseline CNN, a deeper custom CNN with modern regularization techniques, and a MobileNetV2 model leveraging transfer learning from ImageNet. Through this progression, we evaluate how architectural depth, regularization, and pretrained feature representations each contribute to performance on a challenging fine-grained recognition task.

---

## 2. Data Overview

### 2.1 Source and Structure

The Food-101 dataset was loaded from Hugging Face using the `datasets` library. The dataset contains **101,000 images** spanning **101 food categories** with exactly **1,000 images per class**, making it perfectly balanced (imbalance ratio = 1.00). Categories include diverse dishes such as pad thai, crème brûlée, grilled cheese sandwich, beignets, and steak. Each sample consists of a PIL image and an integer class label.

### 2.2 Key Characteristics

**Image variability.** The images are web-scraped photographs with significant variation in resolution, lighting, color balance, camera angle, and composition. Image widths and heights range from approximately 287 to 512+ pixels, with the majority clustering around 384–512 pixels. All images are in color, though some edge cases required explicit RGB conversion.

**Class balance.** The dataset is uniformly distributed with 1,000 samples per class. This means accuracy is a valid primary metric and no rebalancing techniques (e.g., class weighting, oversampling) are required.

**Fine-grained similarity.** Many food categories are visually similar. For example, steak and filet mignon share similar colors and textures; chocolate cake and chocolate mousse have nearly identical color palettes; ice cream and frozen yogurt are often presented in similar bowls. This inter-class similarity is the primary modeling challenge.

**Data quality.** A corruption scan of 2,000 randomly sampled images found zero corrupted or unreadable files, confirming high data quality.

### 2.3 Train/Validation/Test Splits

The dataset was split into **80% training / 10% validation / 10% test** using scikit-learn's `train_test_split` with `random_state=42`. The split was performed in two stages: first an 80/20 train/temporary split, then a 50/50 validation/test split of the temporary set. Both stages used stratified splitting to ensure proportional class representation across all subsets, yielding approximately 800 training, 100 validation, and 100 test images per class. Since Food-101 images are unique photographs with no overlapping subjects, data leakage is not a concern.

---

## 3. Modeling Approach

We trained three models of increasing complexity to understand how each architectural decision contributes to performance.

### 3.1 Model 1: Baseline CNN

The baseline is a minimal Sequential CNN designed to establish a lower bound on performance:

```
Conv2D(32, 3×3, relu) → MaxPool(2×2) → Conv2D(64, 3×3, relu) → MaxPool(2×2)
→ Flatten → Dense(128, relu) → Dense(101, softmax)
```

This architecture provides basic spatial feature extraction through two convolutional layers. The `Flatten` layer feeds directly into a dense classifier. The model contains approximately **23.9 million parameters**, most of which reside in the large Flatten-to-Dense connection — an inherently inefficient design that served as a deliberate point of comparison.

### 3.2 Model 2: Custom CNN with BatchNorm and Dropout

The custom model extends the baseline with four key improvements motivated by the baseline's poor performance:

```
[Conv2D(32) → BatchNorm → ReLU → MaxPool] →
[Conv2D(64) → BatchNorm → ReLU → MaxPool] →
[Conv2D(128) → BatchNorm → ReLU → MaxPool] →
[Conv2D(256) → BatchNorm → ReLU → MaxPool] →
GlobalAveragePooling2D → Dropout → Dense(101, softmax)
```

**Deeper feature hierarchy:** Four convolutional blocks (up from two) allow the network to learn progressively more abstract features — from edges and textures to complex food-specific patterns.

**Batch Normalization:** Added after each convolution to stabilize gradient flow and enable higher learning rates, addressing the training instability observed in the baseline.

**Dropout:** Applied before the final dense layer to reduce overfitting by randomly deactivating neurons during training.

**GlobalAveragePooling2D:** Replaces `Flatten`, reducing the parameter count to just **482K** — a 50× reduction from the baseline — while maintaining spatial information.

### 3.3 Model 3: MobileNetV2 with Transfer Learning

The transfer learning model uses MobileNetV2 pretrained on ImageNet as a feature extractor, with a custom classification head:

```
MobileNetV2 (pretrained, ImageNet) → GlobalAveragePooling2D →
Dense(256, relu) → Dropout(0.3) → Dense(101, softmax)
```

MobileNetV2 was selected for its efficiency (depthwise separable convolutions), competitive ImageNet accuracy, and availability in `tf.keras.applications`. The model has approximately **2.6 million total parameters**, sitting between the baseline and custom CNN in size while dramatically outperforming both.

---

## 4. Training Strategy

### 4.1 Preprocessing Pipeline

All models share a common preprocessing pipeline:

1. **Resizing:** All images resized to 224×224 pixels using `tf.image.resize` to provide fixed-size inputs.
2. **Normalization:** Pixel values scaled from [0, 255] to [0, 1] (for baseline and custom CNN) or preprocessed with `tf.keras.applications.mobilenet_v2.preprocess_input` (for MobileNetV2) to match the pretrained model's expected input distribution.
3. **RGB conversion:** All images explicitly converted to RGB.
4. **Data augmentation** (training only): Random horizontal flips, brightness jitter, and contrast jitter (0.85–1.15×) to increase effective training diversity and build robustness to the lighting and color variation present in the dataset.

Data pipelines were built using `tf.data.Dataset` with prefetching (`AUTOTUNE`) for efficient GPU utilization, with a batch size of 32.

### 4.2 Loss Function and Optimizer

All three models use **sparse categorical crossentropy** as the loss function, which is appropriate for multi-class single-label classification with integer-encoded labels. The **Adam optimizer** was used for all models, with learning rates adjusted per model and training phase.

### 4.3 Training Schedules

**Baseline and Custom CNN:** Trained with a learning rate of 1e-3 using early stopping based on validation loss to prevent unnecessary computation.

**MobileNetV2 — Two-Phase Fine-Tuning:**

- **Phase 1 (Feature extraction):** The entire MobileNetV2 base was frozen and only the classification head was trained with a learning rate of **1e-3**. This allows the head to learn how to map pretrained ImageNet features to the 101 food classes without disturbing the base weights.
- **Phase 2 (Fine-tuning):** The top 30 layers of MobileNetV2 were unfrozen and training continued with a reduced learning rate of **1e-4**. The lower learning rate prevents catastrophic forgetting of pretrained knowledge while allowing the upper convolutional layers to adapt their representations to food-specific characteristics. Bottom layers were kept frozen because they capture universal low-level features (edges, textures) that transfer well across domains.

### 4.4 Reproducibility

All experiments used a fixed random seed of 42 across Python, NumPy, and TensorFlow to ensure reproducible results.

---

## 5. Evaluation

### 5.1 Metrics

Given the perfectly balanced class distribution, we used the following metrics:

- **Top-1 Accuracy** (primary): The fraction of correctly classified images. Valid as the primary metric because every class contributes equally.
- **Macro-averaged F1 Score** (secondary): Computes F1 per class and averages equally, providing a complementary view that ensures performance on every class matters.
- **Per-class Precision and Recall:** To identify which specific food categories the model struggles with.
- **Confusion Matrix:** To visualize systematic misclassification patterns between visually similar categories.

### 5.2 Training Curves

The training curves across models tell a clear story of progression:

**Baseline CNN:** Validation accuracy barely reaches ~10.7%, with high and unstable validation loss throughout training. The primary problem is insufficient model capacity rather than overfitting — the architecture is simply too shallow to learn meaningful food features.

**Custom CNN:** Shows a notably smaller gap between training and validation accuracy compared to the baseline, confirming that BatchNorm and Dropout effectively reduce overfitting. The validation loss curve is more stable. Accuracy reaches approximately 35–45%.

**MobileNetV2:** Displays a distinctive two-phase pattern. During Phase 1, validation accuracy rises quickly as the classification head leverages ImageNet features. During Phase 2, accuracy increases further as the upper layers adapt to food-specific features, reaching approximately 67% validation accuracy.

### 5.3 Visualizations

Confusion matrix analysis of the best model (MobileNetV2) reveals that misclassifications concentrate in semantically meaningful clusters. The most confused class pairs share nearly identical visual characteristics: similar colors, textures, plating styles, and shapes. The 5 hardest classes include pork chop, filet mignon, apple pie, chocolate mousse, and bread pudding. Conversely, the easiest classes are visually distinctive: edamame, onion rings, spaghetti carbonara, french fries, and pizza.

---

## 6. Key Results and Analysis

### 6.1 Summary Table

| Model               | Val Accuracy | Test Accuracy | Test F1 (Macro) | Parameters | Train Time (min) |
|---------------------|-------------|---------------|-----------------|------------|-------------------|
| Baseline CNN        | 10.7%       | 11.1%         | 0.087           | 23.9M      | Fastest            |
| Custom CNN          | 39.5%       | 38.7%         | ~0.38           | 482K       | 69.0               |
| MobileNetV2         | 66.8%       | 65.5%         | 0.655           | 2.6M       | 44.9               |

### 6.2 Analysis

**Depth and regularization matter, but pretrained features dominate.** The custom CNN's fourfold accuracy improvement over the baseline (11% → 39%) demonstrates that architectural depth, BatchNorm, and Dropout provide substantial gains. However, the jump from the custom CNN to MobileNetV2 (39% → 66%) shows that pretrained feature representations are the single most impactful factor for fine-grained recognition.

**Parameter count does not predict performance.** The baseline has the most parameters (23.9M) yet the worst accuracy, because most parameters are wasted in the Flatten-to-Dense connection. MobileNetV2 achieves the best accuracy with a moderate 2.6M parameters thanks to efficient depthwise separable convolutions and pretrained weights.

**Transfer learning improves both accuracy and efficiency.** MobileNetV2 not only achieves the highest accuracy but also trains faster than the custom CNN (44.9 vs. 69.0 minutes). Phase 1 training is fast because only the classification head updates; the forward pass through the frozen base is just inference.

**Error patterns reflect genuine visual similarity.** The remaining misclassifications are concentrated among visually similar food pairs (e.g., steak vs. filet mignon, chocolate cake vs. chocolate mousse). These errors reflect genuine ambiguity in the images rather than model failures, suggesting that further gains will require either higher-resolution inputs, additional context, or more sophisticated architectures.

---

## 7. Limitations and Future Work

### 7.1 Limitations

**Accuracy ceiling.** While 65.5% test accuracy represents a large improvement over training from scratch, it remains below state-of-the-art results on Food-101 (which exceed 90% with larger models and more extensive fine-tuning). The gap indicates room for improvement in both architecture selection and training strategy.

**Limited augmentation.** Our augmentation pipeline included only horizontal flips, brightness, and contrast jitter. More aggressive augmentations (rotation, random cropping, cutout/random erasing) could further improve generalization.

**Single pretrained backbone.** We evaluated only MobileNetV2. Heavier architectures such as EfficientNet or ResNet50 may yield higher accuracy at the cost of increased training time and memory.

**Fixed resolution.** All images were resized to 224×224, which discards fine details that may be important for distinguishing visually similar categories. Higher resolutions (e.g., 299×299 or 384×384) could help but would increase computational cost.

**Training environment constraints.** All training was conducted on a single GPU (NVIDIA GeForce RTX 3090), which limited batch sizes and the number of hyperparameter configurations we could explore.

### 7.2 Future Work

1. **More aggressive fine-tuning:** Unfreeze more layers of the base model and experiment with discriminative learning rates (lower rates for early layers, higher for later layers) to adapt features more deeply without catastrophic forgetting.

2. **Stronger data augmentation:** Add random rotation, zoom/crop, cutout, and mixup to further regularize training and increase effective dataset diversity.

3. **Learning rate scheduling:** Implement cosine annealing or reduce-on-plateau scheduling to improve convergence during fine-tuning.

4. **Alternative architectures:** Evaluate EfficientNetB0/B3 or ResNet50 as alternative backbones, which may offer better feature representations for fine-grained food classification.

5. **Higher input resolution:** Experiment with 299×299 or larger inputs to preserve fine-grained visual details that are lost at 224×224.

6. **Class hierarchy exploitation:** Group visually similar classes and apply hierarchical classification to reduce confusion among the hardest category pairs.

---

## References

- Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101 — Mining Discriminative Components with Random Forests. *European Conference on Computer Vision (ECCV).*
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR.*
- Hugging Face Datasets Library. https://huggingface.co/docs/datasets
- TensorFlow/Keras Documentation. https://www.tensorflow.org/api_docs
