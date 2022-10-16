using Microsoft.ML.Trainers;
using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class LdSvmTrainer : TrainerBase<LdSvmModelParameters>
    {
        public LdSvmTrainer(int treeDepth) : base()
        {
            Name = $"Ld Svm-{treeDepth}";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .LdSvm(treeDepth: treeDepth);
        }
    }
}
