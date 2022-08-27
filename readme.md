# imgcap
### Image Captioning model with weights included, ready for inference out-of-the-box, with sample client implementation 

- Use the included [deploy.sh](deploy.sh) script to host a ready for inference container
- Two variants available for inference, both having accuracy/inference time trade offs  (/imgcap/predict/v1 or /imgcap/predict/v2)
  
 **Related paper** : [Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." computer vision and pattern recognition (2015): 3156-3164.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)

**Dataset used for training**: [Flickr8k](https://forms.illinois.edu/sec/1713398)

**Inspired by** :  [Show-And-Tell-Keras](https://github.com/soloist97/Show-And-Tell-Keras)

**Usage** :
- Run [deploy.sh](deploy.sh) script
- Change existing server config with yours in [imgcap_sample_client.ipynb](client/imgcap_sample_client.ipynb)
- Enjoy 👍
