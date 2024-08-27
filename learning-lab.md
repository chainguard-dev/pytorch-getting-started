
# AI Images Learning Lab Commands

The following commands will allow you to follow along during the Learning Lab on Chainguard's AI Images. For more detailed explanation, check out the [Getting Started tutorial for the pytorch image](https://edu.chainguard.dev/chainguard/chainguard-images/getting-started/pytorch/) or the [Securing the AI/ML Supply Chain ](https://courses.chainguard.dev/securing-ai) course.

## Training the Model

The following command will create a new folder called image_classification in your home folder, change the working directory to that folder, download an archive containing the training data and script and extract it, run the PyTorch Chainguard Image, and execute the training script inside the container.

Before running, consider reviewing the following files that will be downloaded:

- [PyTorch Training script](https://github.com/chainguard-dev/pytorch-getting-started/blob/main/image_classification.py)
- [Training data](https://github.com/chainguard-dev/pytorch-getting-started/tree/main/data/octopus-penguin-whale/train)
- [Validation data](https://github.com/chainguard-dev/pytorch-getting-started/tree/main/data/octopus-penguin-whale/val)


Training command:

```bash
mkdir -p ~/image_classification && cd ~/image_classification && \
curl https://codeload.github.com/chainguard-dev/pytorch-getting-started/tar.gz/main | \
 tar -xz --strip=1 pytorch-getting-started-main/ && \
 docker run --user root --rm -it \
 --platform linux/amd64 \
 -v "$PWD/:/home/nonroot/octopus-detector" \
 cgr.dev/chainguard/pytorch:latest \
 "/home/nonroot/octopus-detector/image_classification.py"
```

After running this command, there should be an `octopus_whale_penguin_model.pt` pickled model file in your `~/image_classification` folder on your host machine.

## Running in Inference

We will now run the model in inference to make a determination on whether a specific image is of an octopus, whale, or penguin.

First, you'll need to download an image. 

```bash
curl https://raw.githubusercontent.com/chainguard-dev/pytorch-getting-started/main/inference-images/octopus.jpg > ~/image_classification/octopus.jpg
```

Run in inference to classify the image:

```bash
cd ~/image_classification && \
 docker run --user root --rm -it \
 --platform linux/amd64 \
 -v "$PWD:/home/nonroot/octopus-detector" \
 cgr.dev/chainguard/pytorch:latest \
 "/home/nonroot/octopus-detector/image_classification.py" \
 "/home/nonroot/octopus-detector/octopus.jpg"
```
