## Why does raw downloaded dataset have more samples than in the constants.py?
E.g. outdoor_day1 has 5133, but the constants shows 5125.

Reason: No Clue, the dataset validation script also has bugs and does not really check the amount, rather it checks whether the files exists (5125 existing files)