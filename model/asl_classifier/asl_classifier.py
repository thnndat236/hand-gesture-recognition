import numpy as np
from ai_edge_litert.interpreter import Interpreter

ASL_TFLITE_MODEL = 'model/asl_classifier/asl_classifier.tflite'

class ASLClassifier(object):
    def __init__(self, model_path=ASL_TFLITE_MODEL, num_thread=1):
        self.interpreter = Interpreter(model_path=model_path,
                                               num_threads=num_thread)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def __call__(self, landmark_list):
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            np.array([landmark_list], dtype=np.float32)
        )

        self.interpreter.invoke()

        result = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )

        result_index = np.argmax(np.squeeze(result))

        return result_index

# def main():
#     keypoint_classifier = ASLClassifier()
#     result = keypoint_classifier([0.0,0.0,0.20078740157480315,-0.051181102362204724,0.3661417322834646,-0.18110236220472442,0.484251968503937,-0.30708661417322836,0.594488188976378,-0.38188976377952755,0.2637795275590551,-0.46062992125984253,0.3425196850393701,-0.65748031496063,0.3937007874015748,-0.7834645669291339,0.42913385826771655,-0.9015748031496063,0.14960629921259844,-0.5,0.1968503937007874,-0.7244094488188977,0.23228346456692914,-0.8740157480314961,0.2559055118110236,-1.0,0.03543307086614173,-0.4881889763779528,0.031496062992125984,-0.7047244094488189,0.03543307086614173,-0.8503937007874016,0.03937007874015748,-0.9763779527559056,-0.07480314960629922,-0.42913385826771655,-0.14566929133858267,-0.5787401574803149,-0.18503937007874016,-0.6850393700787402,-0.2204724409448819,-0.7874015748031497])
#     print(f"result: {result}")

# if __name__ == "__main__":
#     main()