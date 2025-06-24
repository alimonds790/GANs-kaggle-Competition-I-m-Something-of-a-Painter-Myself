# GANs-kaggle-Competition-I-m-Something-of-a-Painter-Myself

## Author: Ali Saleh
## Notebook link since I failed to upload normally due to size: https://colab.research.google.com/drive/1k-RMUdB1F-Yd02o3mD-02V1TU_vadifv?usp=drive_link
This notebook presents a solution for the Kaggle "I'm Something of a Painter Myself" competition. The objective is to construct a Generative Adversarial Network (GAN) capable of generating 7,000 to 10,000 new images in the distinctive artistic style of Claude Monet.

## The approach involves two competing neural networks:
1.  **A Generator Model:** Takes random noise as input and learns to upsample it into full-color images that emulate Monet's work.
2.  **A Discriminator Model:** A classification network trained to distinguish between authentic Monet paintings from the dataset and the synthetic images produced by the Generator.

Through this adversarial training process, the Generator is progressively refined to produce more realistic and stylistically accurate images. This notebook will detail the data processing pipeline, the architecture for both models, the training loop implementation, and a final look at the quality of the generated images.

**This model was trained using L4 GPU on Colab in Batches of 15.**

## Discriminator Overview

This is a **PatchGAN** discriminator. Its job is not to classify the *entire* image as real or fake with a single score. Instead, it classifies overlapping **patches** (regions) of the image as real or fake. This provides richer, more localized feedback to the generator, telling it *which parts* of the image need improvement.

### Components:

*   **`Conv2D` Layers:** These are the core feature extractors. The `strides=(2, 2)` systematically reduces the image size while increasing the number of filters, allowing the network to learn increasingly complex and abstract features—from simple edges to textures and shapes.

*   **`LeakyReLU`:** This activation function is used instead of a standard ReLU because it allows a small gradient to flow even for negative inputs. This prevents "dying neurons" and helps maintain a healthier, more stable training process, which is critical for GANs.

*   **`BatchNormalization`:** This layer stabilizes training by normalizing the activations between layers. It helps prevent the model's internal values from becoming too large or small, leading to faster and more reliable convergence.

*   **`SpectralNormalization` (Most Important):** This is a powerful regularization technique that is key to stable GAN training. It constrains the weights of each layer, which prevents the discriminator from becoming too powerful too quickly. This ensures the generator always receives a smooth and useful gradient to learn from, preventing common training failures like mode collapse.

*   **Final `Conv2D` (Patch Output):** This final layer with a single filter produces the 2D "patch" output grid (e.g., 32x32x1). Each point in this grid corresponds to a verdict on a specific patch of the original input image, completing the PatchGAN architecture.

## Generator Overview

This is a **ResNet-style Generator**. Its purpose is to act as the "artist," transforming a random noise vector (the `latent_dim`) into a full-color, Monet-style image. It builds the image progressively, starting from a tiny 4x4 feature map and scaling it up to the final 256x256 size.

### Components:

*   **Input (Latent Vector):** A random vector that serves as a unique "seed" or source of inspiration for each generated painting.

*   **`Dense` and `Reshape` Layers:** These initial layers transform the 1D input vector into a small, 3D feature map (4x4x512), creating the initial "canvas" for the generator to work on.

*   **Residual Blocks (`res_block_g_v4`):** The core of the network. Each block upscales the image and refines details. The key feature is the **`Skip Connection`**, which adds the block's input to its output. This dramatically improves training stability and allows the network to learn complex features more easily.

*   **`UpSampling2D` and `Conv2D`:** These are the tools for creation. `UpSampling2D` doubles the image size at each step, while `Conv2D` layers act like paintbrushes, learning to add textures, colors, and shapes.

*   **`BatchNormalization` and `LeakyReLU`:** Just like in the discriminator, these are critical for stabilizing the training process and ensuring the network learns efficiently without getting stuck.

*   **Final `Conv2D` with `tanh`:** The final layer that renders the image. The **`tanh`** activation squashes pixel values into the standard [-1, 1] range, which is the expected format for GAN training.

*   **`SpectralNormalization` (Uncommon but included):** While typically used in discriminators, it's added here to the first and last layers as an extra measure to further stabilize the complex training dynamics between this powerful generator and its discriminator.

## Training Configuration

*   **`epochs = 300`:** The model will iterate through the entire dataset 300 times as GANs often require many epochs to produce high-quality results.

*   **`Adam` Optimizers:** We use two separate optimizers because the generator and discriminator learn independently.
    *   **Different Learning Rates:** Notice the generator's learning rate (`1e-4`) is ten times higher than the discriminator's (`1e-5`). This is a common and important trick in GAN training. It slightly "handicaps" the discriminator, preventing it from overpowering the generator too early and giving the generator a better chance to learn.
    *   **`beta_1=0.5`:** This parameter is adjusted from the default of 0.9 to help stabilize the often-oscillating training dynamics of GANs.
    *   **`clipnorm=1.0`:** This is **gradient clipping**, a safety measure that prevents the updates to the models from becoming too large and erratic, which could destabilize the entire training process.

*   **`loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)`:** This is the standard loss function for a real vs. fake classification task. `from_logits=True` is a critical and numerically stable choice, since we used a tanh activation function

# Conclusion

In this project, we successfully built and trained a Generative Adversarial Network (GAN) to create new paintings in the style of Claude Monet.

We used a "Painter" model (the Generator) to create the art and a "Critic" model (the Discriminator) to judge it. By making them compete, the Painter learned to create images that capture the colors, textures, and overall feeling of Monet's work. This shows that GANs can be a powerful and fun tool for creating art.

# Future Improvements

Here are a few ideas to make the model even better:

*   **Train for Longer:** Running the training for more epochs (e.g., 500 instead of 300) might lead to more detailed and polished images.

*   **Tune the Learning Rates:** We can experiment with slightly different learning rates for the Painter and the Critic to see if we can find a better balance between them.

*   **Try a Different Model:** We could use a more advanced model design like **StyleGAN2**, which is famous for creating extremely high-quality and realistic images.

*   **Turn Photos into Paintings:** We could build a **CycleGAN**, a different type of model that can learn to turn any regular photograph into a Monet-style painting.

# References:

* Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. In *Advances in Neural Information Processing Systems, 27*. Curran Associates, Inc.

* Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). Analyzing and improving the image quality of StyleGAN. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 8110-8119).

* Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised representation learning with deep convolutional generative adversarial networks*. arXiv preprint arXiv:1511.06434.

* Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 2223-2232).
