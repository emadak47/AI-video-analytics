# AI-Video-Analytics

## Description 
This project is designed to analyze candidates' facial expressions, eye gaze, emotions, and audio during a one-way interview scenario like HireVue. The recorded video, along with the analysis, is submitted to the respective institution for further evaluation.

The project offers statistics on the aforementioned metrics and holistic metrics such as attentiveness and confidence. The system operates in two modes, including static and real-time analysis. The user can upload a pre-recorded video or perform a live analysis using a simple integrated front-end interface.

In the case of live analysis, some statistics will be available during the process, while others will require the feed to end before further processing. The primary aim of this project is to provide detailed and comprehensive evaluation of candidates to facilitate the recruitment process.

### Note
- It is worth noting that this project is solely a graduation project assigned to me for my undergraduate degree. 
- While it provides an in-depth evaluation of candidates, I must clarify that I do not endorse its use. In my opinion, the implementation of such technology may further widen the gap between candidates and recruiters, which is an ever-growing issue in the hiring process.

## Set-up
- python >= v 3.9 
- install `pipenv` >= v 2022.6.7 
- create a new virtual env. 
- activate a new shell in the working directory    
- install all necessary packages in `Pipfile.lock`

## Run
1. Spawn a new shell in your vitualenv
``` 
pipenv shell 
```

2. Run backend
``` 
python backend.py 
```

3. Open a new browser tab and navigate to local host port 5000. 

4. Click on the respective buttons for analysis.


## Potential Improvements 
- `real_time` and `static` clients have repetitive codes. Better to create a parent client (class) where both clients can inherent abstract methods and modify their local definitions accordingly. 

- Cross database validation and testing for all models. 

- Use a more solid framework for front-end, i.e. React. 