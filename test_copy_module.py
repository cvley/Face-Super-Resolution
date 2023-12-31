import bmf
import sys

input_file = sys.argv[1]
output_path = 'copy.mp4'

(
    bmf.graph()
       .decode({'input_path': input_file})['video']
       .module('face_sr_module')
       .encode(None, {"output_path": output_path})
       .run()
)

