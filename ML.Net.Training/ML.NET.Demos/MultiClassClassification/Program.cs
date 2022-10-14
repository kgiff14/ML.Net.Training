using MultiClassClassification.MachineLearning.Common;
using MultiClassClassification.MachineLearning.DataModels;
using MultiClassClassification.MachineLearning.Predictors;
using MultiClassClassification.MachineLearning.Trainers;

namespace MultiClassClassification
{
    public class Program
    {
        static void Main(string[] args)
        {
            var newSample = new PenguinData
            {
                Island = "Torgersen",
                CulmenLength = 18.7f,
                CulmenDepth = 39.3f,
                FilperLength = 180,
                BodyMass = 3700,
                Sex = "MALE"
            };

            var trainers = new List<ITrainerBase>
            {
                new LbfgsMaximumEntropyTrainer(),
                new NaiveBayesTrainer(),
                new OneVersusAllTrainer(),
                new SdcaMaximumEntropyTrainer(),
                new SdcaNonCalibratedTrainer()
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

            Console.WriteLine($"\nMacro Accuracy: {modelMetrics.MacroAccuracy: #.##}");
            Console.WriteLine($"\nMicro Accuracy: {modelMetrics.MicroAccuracy: #.##}");
            Console.WriteLine($"\nLog Loss: {modelMetrics.LogLoss: #.##}");
            Console.WriteLine($"\nLog Loss Reduction: {modelMetrics.LogLossReduction: #.##}");

            trainer.Save();

            var predictor = new Predictor();
            var prediction = predictor.Predict(sample);
            Console.WriteLine($"\nPrediction: {prediction.PredictedLabel:#.##}");
        }
    }
}
