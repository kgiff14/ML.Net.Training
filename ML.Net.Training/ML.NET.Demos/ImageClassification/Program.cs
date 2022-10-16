using ImageClassification;
using ImageClassification.MachineLearning;

var dataLoader = new DataLoader(@"D:\ML.Net.Training\ML.Net.Training\ML.NET.Demos\ImageClassification\Data\training_set\");
var trainer = new ImageClassifier(Microsoft.ML.Vision.ImageClassificationTrainer.Architecture.MobilenetV2, dataLoader);
trainer.Fit();
trainer.Save();

var predictor = new Predictor(dataLoader);

Console.WriteLine("Make predictions on test dataset:");
var predictions = predictor.MakeTestDatasetPredictions();

foreach (var prediction in predictions)
{
    string imageName = Path.GetFileName(prediction.ImagePath);
    Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
}