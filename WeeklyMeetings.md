# Weekly Meetings Markdown

## Week 37 Meeting 15th September
- Who did you help this week?
    - I helped my colleagues find better annotation guidelines. 
- What helped you this week?
    - Talking to my colleagues about my concerns surrounding the annotation guidelines, finding a better solution
- What did you achieve?
    - I got a new annotation guideline for the 2023 data confirmed. We will now take the videos and then extract every 30 sec from them, and then only annotate those.
    - I have started implementing that and extracting the frames from the 2023 data
- What did you struggle with?
    - Masking the data for 30 second intervals and then trying to extract them was unsuccessful, instead I am using all the annotated data points to extract frames and then afterwards trying to get a frame around every 30 sec (if they exist)
- What would you like to work on next week?
    - Implementing the Label Studio setup.
    - Improving the OCR, trying to modify the image in such a way that it works better
- Where do you need help from Veronika?
    - Tips on improving the OCR
    - General discussion surrounding the Confidentiality Agreement, project roadmap, what I should/shouldn't focus on
    - Do I need to submit a project statement? 
- What are the agreements after this meeting? (to fill in after the meeting)

## Week 38 Meeting 19th September (Email)
- Who did you help this week?
    - N/A
- What helped you this week?
    - The LabelStudio repo helped me figure out how to deploy Label Studio to github
- What did you achieve?
    - I deployed Label Studio to github
    - I setup azure containers which hold the annotation data for Label Studio
    - I processed and prepared the R1 2023 Transect data for annotation, masked it for and information which could cause annotation bias
- What did you struggle with?
    - It wasn't quite as easy to setup as I thought, took some figuring details out
- What would you like to work on next?
    - Preparing the Machine Learning Part
- Where do you need help from Veronika?
    - I am unsure about the annotation setup. The output values are in bins which are unevenly sized
- What are the agreements after this meeting? (to fill in after the meeting)
    - Label Studio is good
    - I can use the binning setup

## Week 39 Meeting 26th September (Email)
- Who did you help this week?
    - N/A
- What helped you this week?
    - Talking to the Marine Biologists, figuring out what they want as annotation guidelines and why
- What did you achieve?
    - Changed the annotation setup to percentage values
    - Finalised the Label Studio projects for each Marine Biologist
- What did you struggle with?
    - The annotation guidelines of the marine biologists, not quite easy to work with since its not really correct
- What would you like to work on next?
    - Preparing the machine learning, finalising the preprocessing
    - Work with the person who ingests the data into the coverage models
    - Get the data from the PHD student who did a similar project at the company
    - Preparing the analysis to compare their video annotation and the label studio annotation
- Where do you need help from Veronika?
    - Don't think I really needed any help this week which I guess is a good thing
- What are the agreements after this meeting? (to fill in after the meeting)
    - Add the emails as weekly meeting notes (Now complete :) )

## Week 40 Meeting 6th October
- Cancelled due to conference

## Week 41 Meeting 13th October
- Who did you help this week?
    - N/A Is it a bad thing that I don't necessarily help anyone?...
    - Actually no I did help Jorge with figuring out the confidentiality agreement stuff for his company collaboration project
- What helped you this week?
    - I moved the excel/csv annotation data to an sql database which has helped me organise my data a lot
- What did you achieve?
    - Moved the annotation data to an azure sql database
    - Changed the Label Studio data again since I accidentally uploaded the wrong data...
- What did you struggle with?
    - Some of the data is missing columns, eg: Date, Longitude. This means I can't process all the data
    - Just in general my repo has gotten pretty messy so the azure sql database should help with this
- What would you like to work on next?
    - Preprocessing all the videos into frames with values
    - Will then move the final images (which are masked) to azure storage blob containers
- Where do you need help from Veronika?
    - None needed this week, just a general update on how things are going
- What are the agreements after this meeting? (to fill in after the meeting)

## Week 42
- (Break)

## Week 43
- (No meeting, was busy)

## Week 44 Meeting 30th October
- Who did you help this week?
    - N/A
- What helped you this week?
    - N/A hard work and grinding through stupid mistakes
- What did you achieve?
    - Dataset is done!!! :) 
- What did you struggle with?
    - My Homeserver and VM broke and I didn't commit or push any changes so I had to redo 3 weeks of work..... :|
- What would you like to work on next?
    - Machine Learning!
- Where do you need help from Veronika?
    - Ideas about Machine Learning, where to start from
- What are the agreements after this meeting? (to fill in after the meeting)
    - Try using vesselness filters 
    - Setup a two-step ML model, with one head for classification and one head for regression
    - Do timeseries training

## Week 44 Meeting 3rd November
- Who did you help this week?
    - N/A
- What helped you this week?
    - Pytorch documentation
- What did you achieve?
    - Setup the first ML experiments, tested different filters and line detection/enhancement filters
- What did you struggle with?
    - Programming a two headed ML model is complex, not quite so simple
- What would you like to work on next?
    - Redoing the two-step model, training it, trying other models
- Where do you need help from Veronika?
    - checking the two-step model architecture is correct
- What are the agreements after this meeting? (to fill in after the meeting)


### Weekly Meeting Template:
- Who did you help this week?
- What helped you this week?
- What did you achieve?
- What did you struggle with?
- What would you like to work on next?
- Where do you need help from Veronika?
- What are the agreements after this meeting? (to fill in after the meeting)
