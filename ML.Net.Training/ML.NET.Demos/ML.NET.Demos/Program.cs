using System.IO;
using BinaryClassification.MachineLearning.Common;
using BinaryClassification.MachineLearning.DataModels;
using BinaryClassification.MachineLearning.Predictors;
using BinaryClassification.MachineLearning.Trainers;

namespace BinaryClassification
{
    public class Program
    {
        static void Main(string[] args)
        {
            var newSample = new PenguinData
            {
                CulmenDepth = 18.4f,
                CulmenLength = 40.8f
            };

            var trainers = new List<ITrainerBase>
            {
                new LbfgsLogisticRegressionTrainer(),
                new AveragedPerceptronTrainer(),
                new PriorTrainer(),
                new SdcaLogicsticRegressionTrainer(),
                new SdcaNonCalibratedTrainer(),
                new SgdCalibratedTrainer(),
                new SgdNonCalibratedTrainer()
            };

            trainers.ForEach(x => TrainEvalutePredict(x, newSample));
        }

        static void TrainEvalutePredict(ITrainerBase trainer, PenguinData sample)
        {
            Console.WriteLine($"\n\n-------------------------------------------\n\t{trainer.Name}\n-------------------------------------------");

            string fileName = "penguins_size_binary.csv";
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
            Console.WriteLine($"\nPrediction: {prediction.PredictedLabel:#.##}");
        }
    }
}
