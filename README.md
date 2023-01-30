# Waveform Synthesizer with Diffusion Models (WIP)
(Self taught leanring materials) Diffusion based waveforom synthesizer. A collection or (re)implementations of interesting audio related diffusion models. Trainging framework is based on the [Diffwave repo](https://github.com/lmnt-com/diffwave). 

# TODO
- [ ] Unconditional diffusion
- [ ] Unconditional diffusion w and w/o guidance
- [ ] Conditional diffusion: Encoder(Conditioner) + Decoder(Vocoder)
- [ ] Notes

### Evaulation of Unconditional Generators
We compare different frameworks by testing on sc09 dataset using [unconditional audio generation benchmark repo](https://github.com/gzhu06/Unconditional-Audio-Generation-Benchmark).

| System    |  Backbone   | Sampler |FID | Inception  | mInception | AM | 
|-----------|------------|-----------|-------|---------|--------|--------|
|[diffwave](https://github.com/philsyn/DiffWave-unconditional)  | WaveNet | AS|1.80|5.70|51.88|0.65|
|[audiodiff](https://github.com/archinetai/audio-diffusion-pytorch)  | UNet1d | AS|1.51|7.07|105.8|0.471|


## References
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Code for Unconditional Diffwave](https://github.com/philsyn/DiffWave-unconditional)
- [Code for Audio Diffusion](https://github.com/archinetai/audio-diffusion-pytorch)
