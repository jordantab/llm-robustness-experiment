PROMPT_SET = {
    'sst2': [
        'Is the following sentence positive or negative? Answer me with "positive" or "negative", just one word. ',
        'Please classify the following sentence into either positive or negative. Answer me with "positive" or "negative", just one word. ',
    ],
    'qqp': [
        'Are the following two questions equivalent or not? Answer me with "equivalent" or "not_equivalent". ',
    ],
    'mnli': [
        'Are the following two sentences entailment, neutral or contradiction? Answer me with "entailment", "neutral" or "contradiction". ',
    ],
    'anli': [
        'Are the following paragraph entailment, neutral or contradiction? Answer me with "entailment", "neutral" or "contradiction". The answer should be a single word. The answer is: ',
    ],
    'qnli': [
        'Are the following question and sentence entailment or not_entailment? Answer me with "entailment" or "not_entailment". ',
    ],
    'mnli-mm': [
        'Are the following two sentences entailment, neutral or contradiction? Answer me with "entailment", "neutral" or "contradiction". ',
    ],
    'flipkart': [
        'Is the following sentence positive, neutral, or negative? Answer me with "positive", "neutral", or "negative", just one word. ',
    ],
    'rte': [
        'Are the following two sentences entailment or not_entailment? Answer me with "entailment" or "not_entailment". ',
    ],
    'ddxplus': [
        "Imagine you are a doctor. Based on the dialogue, what is the diagnosis? \
        Only select one answer among the diseases: spontaneous pneumothorax, cluster headache, boerhaave, spontaneous rib fracture, gerd, hiv (initial infection), anemia, viral pharyngitis, inguinal hernia, myasthenia gravis, whooping cough, anaphylaxis, epiglottitis, guillain-barré syndrome, acute laryngitis, croup, psvt, atrial fibrillation, bronchiectasis, allergic sinusitis, chagas, scombroid food poisoning, myocarditis, larygospasm, acute dystonic reactions, localized edema, sle, tuberculosis, unstable angina, stable angina, ebola, acute otitis media, panic attack, bronchospasm / acute asthma exacerbation, bronchitis, acute copd exacerbation / infection, pulmonary embolism, urti, influenza, pneumonia, acute rhinosinusitis, chronic rhinosinusitis, bronchiolitis, pulmonary neoplasm, possible nstemi / stemi, sarcoidosis, pancreatic neoplasm, acute pulmonary edema, pericarditis.\
        Only answer the name of one disease. The answer should just be a single word. The answer is: ",
    ],
    'ddxplus-json': [
        "Imagine you are a doctor. Based on the dialogue, what is the diagnosis? \
        Respond ONLY in JSON with a single field 'disease'.\n\
        The answer can only be one from:\n\
        'spontaneous pneumothorax', 'cluster headache', 'boerhaave', 'spontaneous rib fracture',\
        'gerd', 'hiv (initial infection)', 'anemia', 'viral pharyngitis', 'inguinal hernia', 'myasthenia gravis',\
        'whooping cough', 'anaphylaxis', 'epiglottitis', 'guillain-barré syndrome', 'acute laryngitis', 'croup',\
        'psvt', 'atrial fibrillation', 'bronchiectasis', 'allergic sinusitis', 'chagas', 'scombroid food poisoning',\
        'myocarditis', 'larygospasm', 'acute dystonic reactions', 'localized edema', 'sle', 'tuberculosis',\
        'unstable angina', 'stable angina', 'ebola', 'acute otitis media', 'panic attack', 'bronchospasm / acute asthma exacerbation',\
        'bronchitis', 'acute copd exacerbation / infection', 'pulmonary embolism', 'urti', 'influenza', 'pneumonia',\
        'acute rhinosinusitis', 'chronic rhinosinusitis', 'bronchiolitis', 'pulmonary neoplasm', 'possible nstemi / stemi',\
        'sarcoidosis', 'pancreatic neoplasm', 'acute pulmonary edema', 'pericarditis'.\
        Analyze the prompt and answer without any explanation or additional text.",
    ],
    'translation_en_to_zh': [
        'Translate the following sentence from Engilish to Chinese. '
    ],
    'translation_zh_to_en': [
        'Translate the following sentence from Chinese to English. '
    ]
}

LABEL_SET = {
    'sst2': ['positive', 'negative'],
    'mnli': ['entailment', 'neutral', 'contradiction'],
    'anli': ['entailment', 'neutral', 'contradiction'],
    'qqp': ['equivalent', 'not_equivalent'],
    'qnli': ['entailment', 'not_entailment'],
    'mnli-mm': ['entailment', 'neutral', 'contradiction'],
    'rte': ['entailment', 'not_entailment'],
    'ddxplus': ['spontaneous pneumothorax', 'cluster headache', 'boerhaave', 'spontaneous rib fracture', 'gerd', 'hiv (initial infection)', 'anemia', 'viral pharyngitis', 'inguinal hernia', 'myasthenia gravis', 'whooping cough', 'anaphylaxis', 'epiglottitis', 'guillain-barré syndrome', 'acute laryngitis', 'croup', 'psvt', 'atrial fibrillation', 'bronchiectasis', 'allergic sinusitis', 'chagas', 'scombroid food poisoning', 'myocarditis', 'larygospasm', 'acute dystonic reactions', 'localized edema', 'sle', 'tuberculosis', 'unstable angina', 'stable angina', 'ebola', 'acute otitis media', 'panic attack', 'bronchospasm / acute asthma exacerbation', 'bronchitis', 'acute copd exacerbation / infection', 'pulmonary embolism', 'urti', 'influenza', 'pneumonia', 'acute rhinosinusitis', 'chronic rhinosinusitis', 'bronchiolitis', 'pulmonary neoplasm', 'possible nstemi / stemi', 'sarcoidosis', 'pancreatic neoplasm', 'acute pulmonary edema', 'pericarditis', 'cannot decide'],
    'flipkart': ['positive', 'negative', 'neutral'],

}