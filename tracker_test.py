from trackeranalyzer.tracker_analyzer import TrackerAnalyzer

ma = TrackerAnalyzer('data/data.csv')
ma.analyze_all(save_csv='logs/results.csv')
