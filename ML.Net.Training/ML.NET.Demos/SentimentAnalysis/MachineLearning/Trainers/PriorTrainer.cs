using SentimentAnalysis.MachineLearning.Common;
using Microsoft.ML.Trainers;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class PriorTrainer : TrainerBase<PriorModelParameters>
    {
        public PriorTrainer() : base()
        {
            Name = "Prior";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .Prior(labelColumnName: "Label");
        }
    }
}
