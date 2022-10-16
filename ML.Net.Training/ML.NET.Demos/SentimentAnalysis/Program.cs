using System.IO;
using SentimentAnalysis.MachineLearning.Common;
using SentimentAnalysis.MachineLearning.DataModels;
using SentimentAnalysis.MachineLearning.Predictors;
using SentimentAnalysis.MachineLearning.Trainers;

namespace SentimentAnalysis
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Please enter a sentiment:");
            var result = Console.ReadLine();

            var newSample = new SentimentData
            {
                SentimentText = result
            };

            var trainers = new List<ITrainerBase>
            {
                new LbfgsLogisticRegressionTrainer(),
                new AveragedPerceptronTrainer(),
                new PriorTrainer(),
                new SdcaLogicsticRegressionTrainer(),
                new SdcaNonCalibratedTrainer(),
                new SgdCalibratedTrainer(),
                new SgdNonCalibratedTrainer(),
                new LdSvmTrainer(2),
                new LdSvmTrainer(1),
                new RandomForestTrainer(3,3),
                new RandomForestTrainer(3,8),
                new DecisionTreeTrainer(5,2, 0.5),
                new DecisionTreeTrainer(2,46),
                new GamTrainer()
            };

            trainers.ForEach(x => TrainEvalutePredict(x, newSample));
        }

        static void TrainEvalutePredict(ITrainerBase trainer, SentimentData sample)
        {
            Console.WriteLine($"\n\n-------------------------------------------\n\t{trainer.Name}\n-------------------------------------------");

            string fileName = "imdb_labelled.txt";
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Data\", fileName);

            trainer.Fit(path.ToString());

            var modelMetrics = trainer.Evalute();

            Console.WriteLine($"\nAccuracy: {modelMetrics.Accuracy: 0.##}");
            Console.WriteLine($"\nF1 Score: {modelMetrics.F1Score: #.##}");
            Console.WriteLine($"\nPositive Precision: {modelMetrics.PositivePrecision: #.##}");
            Console.WriteLine($"\nNegative Precision: {modelMetrics.NegativePrecision: 0.##}");
            Console.WriteLine($"\nPositive Recall: {modelMetrics.PositiveRecall: #.##}");
            Console.WriteLine($"\nNegative Recall: {modelMetrics.NegativeRecall: #.##}");
            Console.WriteLine($"\nArea Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve: #.##}");

            trainer.Save();

            var predictor = new Predictor();
            var prediction = predictor.Predict(sample);
            Console.WriteLine($"\nPrediction: {prediction.Prediction:#.##}");
        }
    }
}
