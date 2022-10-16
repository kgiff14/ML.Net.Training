using SentimentAnalysis.MachineLearning.Common;
using Microsoft.ML.Trainers;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class AveragedPerceptronTrainer : TrainerBase<LinearBinaryModelParameters>
    {
        public AveragedPerceptronTrainer() : base()
        {
            Name = "Average Percption";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
