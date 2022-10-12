using DesicionTrees.MachineLearning.Common;
using DesicionTrees.MachineLearning.DataModels;
using DesicionTrees.MachineLearning.Predictors;
using DesicionTrees.MachineLearning.Trainers;

namespace DesicionTrees
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
                new DecisionTreeTrainer(3,3),
                new DecisionTreeTrainer(3,8, 0.1),
                new DecisionTreeTrainer(2,15),
                new DecisionTreeTrainer(5,2, 0.5),
                new DecisionTreeTrainer(2,46),
                new GamTrainer()
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