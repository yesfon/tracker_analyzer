from trackeranalyzer.tracker_analyzer import TrackerAnalyzer

ma = TrackerAnalyzer('data/data.csv')
ma.analyze_all()
ma.save_to_csv(filename='logs/results.csv')
