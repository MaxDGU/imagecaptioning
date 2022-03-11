# imagecaptioning
Image captioning with LSTM generators - captioning 8k flickr images using a conditioned RNN generator. 
Achieves the following: 
* Creates matrices of image representations using an off-the-shelf image encoder.
* Reads and preprocess the image captions. 
* Writes a generator function that returns one training instance (input/output sequence pair) at a time. 
* Trains an LSTM language generator on the caption data.
* Writes a decoder function for the language generator. 
* Adds the image input to write an LSTM caption generator. 
* Implements beam search
