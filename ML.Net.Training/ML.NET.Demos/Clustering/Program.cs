using Clustering.MachineLearning.Common;
using Clustering.MachineLearning.DataModels;
using Clustering.MachineLearning.Predictors;
using Clustering.MachineLearning.Trainers;

namespace Clustering
{
    public class Program
    {
        static void Main(string[] args)
        {
            var newSample = new PenguinData
            {
                Island = "Biscoe",
                CulmenLength = 45.2f,
                CulmenDepth = 14.8f,
                FilperLength = 212,
                BodyMass = 5200,
                Sex = "MALE"
            };

            var trainers = new List<ITrainerBase>
            {
                new KMeansTrainer(5),
                new KMeansTrainer(10),
                new KMeansTrainer(2),
                new KMeansTrainer(9)
            };

            trainers.ForEach(x => TrainEvalutePredict(x, newSample));
        }

        static void TrainEvalutePredict(ITrainerBase trainer, PenguinData sample)
        {
            Console.WriteLine($"\n\n-------------------------------------------\n\t{trainer.Name}\n-------------------------------------------");

            string fileName = "penguins_size.csv";
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Data\", fileName);

            trainer.Fit(path.ToString());

            var modelMetrics = trainer.Evalute();

            Console.WriteLine($"\nAverage Distance: {modelMetrics.AverageDistance: #.##}");
            Console.WriteLine($"\nDavis Bouldin Index: {modelMetrics.DaviesBouldinIndex: #.##}");
            Console.WriteLine($"\nNormalized Mutual Information: {modelMetrics.NormalizedMutualInformation: #.##}");

            trainer.Save();

            var predictor = new Predictor();
            var prediction = predictor.Predict(sample);
            Console.WriteLine($"\nPrediction: {prediction.PredictedLabelId:#.##}");
            Console.WriteLine($"\nDistances: {prediction.Distances:#.##}");
        }
    }
}