# Lab5 MaskGIT for Image Inpainting Experiment Score

> student id: 313551097  
> student name: 鄭淮薰

## Part1: Prove your code implementation is correct

### Show iterative decoding

![gamma.png](img/gamma.png)
- cosine
- linear
- square

#### (a) Mask in latent domain

From the following images, we can observe that the cosine and square methods fill less mask in the early stage, and gradually increase the amount of filling in the later stage, while the linear method maintains a certain amount of filling.

| cosine     | ![cosine mask0.png](img/cosine/mask.png)        |
|------------|-------------------------------------------------|
| **linear** | ![linear mask0.png](img/linear/linear-mask.png) |
| **square** | ![square mask0.png](img/square/square-mask.png) |

#### (b) Predicted image

| cosine     | ![cosine image0.png](img/cosine/image.png) |
|------------|--------------------------------------------|
| **linear** | ![linear image0.png](img/linear/image.png) |
| **square** | ![square image0.png](img/square/image.png) |

## Part2: The Best FID Score

### Screenshot

![best fid](img/best/best-fid.png)

### Masked Images v.s MaskGIT Inpainting Results v.s Ground Truth

| Masked Images                  | ![image_000.png](img/mask/image_000.png) | ![image_002.png](img/mask/image_001.png) | ![image_003.png](img/mask/image_003.png) | ![image_006.png](img/mask/image_006.png) | ![image_008.png](img/mask/image_008.png) | ![image_005.png](img/mask/image_013.png) |
|--------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|
| **MaskGIT Inpainting Results** | ![test0](img/best/image_000.png)         | ![test0](img/best/image_001.png)         | ![test0](img/best/image_003.png)         | ![test0](img/best/image_006.png)         | ![test0](img/best/image_008.png)         | ![test0](img/best/image_013.png)         |
| **Ground Truth**               | ![gt0.png](img/gt/0.png)                 | ![gt2.png](img/gt/1.png)                 | ![gt3.png](img/gt/3.png)                 | ![gt6.png](img/gt/6.png)                 | ![gt8.png](img/gt/8.png)                 | ![gt13.png](img/gt/13.png)               |



### The setting about training strategy, mask scheduling parameters, and so on

- learning rate: 1e-4
- batch size: 10
- epochs: 300
- optimizer: Adam
- sweet spot: 8
- total iteration: 8
- mask function: cosine