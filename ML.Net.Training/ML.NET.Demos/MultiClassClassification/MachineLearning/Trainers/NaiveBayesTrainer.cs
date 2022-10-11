using Microsoft.ML.Trainers;
using MultiClassClassification.MachineLearning.Common;

namespace MultiClassClassification.MachineLearning.Trainers
{
    public class NaiveBayesTrainer : TrainerBase<NaiveBayesMulticlassModelParameters>
    {
        public NaiveBayesTrainer() : base()
        {
            Name = "Naive Bayes";
            Model = MlContext
                        .MulticlassClassification
                        .Trainers
                        .NaiveBayes(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
