# Automated Detection of Defects on Metal Surfaces using Deep Learning Techniques and Vision Transform
This project utilizes machine and deep learning techniques to automate the detection of defects on metal surfaces in industrial products.
The goal of this project is to classify various common metal defects.
The dataset can be found here https://www.kaggle.com/datasets/toqaalaaawad/metal-surfaces-defects  
The table shows the types of defects we're working on
<div align="center">
  <table>
    <tr>
      <td>
        <figure>
          <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/29299c49-f5e5-44be-ac6a-e05fcb997101" alt="Crease"  height =200 width="200"/>
          <div align="center"> <figcaption>Crease</figcaption> </div>
        </figure>
      </td>
      <td>
        <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/714706d8-fa79-4aa9-ba9d-e5abecf07c27 alt="Crescent_gap"  height =200 width="200"/>
        <div align="center"> <figcaption>Crescent gap</figcaption> </div>
        </figure>
      </td>
      <td>
       <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/fcd0be2d-31e3-46fc-b985-9adc4debda45" alt="Inclusion"  height =200 width="200"/>
        <div align="center"> <figcaption>Inclusion</figcaption> </div>
        </figure>
      </td>
      <td>
       <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/e1af9fc7-984c-4215-bc7c-01642ac2abc8" alt="Oil_spot"  height =200 width="200"/>
        <div align="center"> <figcaption>Oil Spot</figcaption> </div>
        </figure>
      </td>
     </tr>
    <tr>
      <td>
        <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/b4461799-cf30-411e-81c3-0d18fe1560a0" alt="Punching_hole"  height =200 width="200"/>
        <div align="center"> <figcaption>Punching hole</figcaption> </div>
        </figure>
      </td>
      <td>
        <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/1d78b86d-5048-4017-97f9-2842710126f3" alt="rolled_in_scale"  height =200 width="200"/>
        <div align="center"> <figcaption>Rolled in Scale</figcaption> </div>
        </figure>
      </td>
      <td>
        <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/0f7f11d0-c5a0-40b7-9ca0-6714991c0588" alt="rolled_pit"  height =200 width="200"/>
        <div align="center"> <figcaption>Rolled pit</figcaption> </div>
        </figure>
      </td>
      <td> 
        <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/61b83e8c-d458-4375-9561-91872af63b39" alt="scratches"  height =200 width="200"/>
        <div align="center"> <figcaption>Scratches</figcaption> </div>
        </figure>
      </td>
     </tr>
    <tr>
      <td> 
         <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/eaf64849-0a30-4bdc-b734-f2223e3266c9" alt="silk spot"  height =200 width="200"/>
        <div align="center"> <figcaption>Silk spot</figcaption> </div>
        </figure>
      </td>
      <td>
        <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/24221c85-7876-4e2a-bd3b-d25ef86aacfb" alt="waist_folding"  height =200 width="200"/>
        <div align="center"> <figcaption>Waist folding</figcaption> </div>
        </figure>
      </td>
      <td>
        <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/d8d7138b-47d9-469b-b5bf-7ace5ab1c7d3" alt="water_spot" height =200 width="200"/>
        <div align="center"> <figcaption>Water spot</figcaption> </div>
        </figure>
      </td>
      <td>
        <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Machine-and-Deep-Learning-Techniques/assets/90696437/5e50bfe1-091a-479e-a674-c5304385368c" alt="welding_line"  height =200 width="200"/>
        <div align="center"> <figcaption>Welding line</figcaption> </div>
        </figure>
      </td>
    </tr>
  </table>
</div>

This repository serves as a comprehensive record of all the steps taken to improve the accuracy of the system.  
                           
## The repository includes the following:
- Categorical classification between two classes using Sigmoid activation.
- Categorical classification among 10 defect classes using a custom CNN model with softmax.
- Categorical classification among 10 defect classes using Inception and SGD.
- Categorical classification among 11 defect classes using a custom CNN model with softmax.
- Categorical classification among 11 defect classes using Inception and SGD.
- Single detection of a defect by training the Inception model on the image annotations.
- Multiple detection of defects by training the Inception model on the image annotations.
- Using VIT as a feature extractor instead of the pretrained model.

# Summary of the most promising results:
                           
                           
# Multiple detection of defects using inception:
## The model architecture:
 <div align="center">

 <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Deep-Learning-Techniques-and-Vision-Transform/assets/90696437/66d980b7-86c0-4ac1-ac2e-d01a6b854b04" alt="Single defect"  height =900 width="700"/>
        <div align="center"> <figcaption>Multiple defect detection model architecture </figcaption> </div>
        </figure>
</div>  
                           
## The model results:
 <div align="center">
 <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Deep-Learning-Techniques-and-Vision-Transform/assets/90696437/f174a669-9b1f-45bc-9f39-5eeb79ce1974" alt="Single defect"  height =400 width="500"/>
        <div align="center"> <figcaption>The accuracy and the validation metrics </figcaption> </div>
        </figure>
</div>          

# Single detection of a defect using inception::
## The model architecture:
 <div align="center">
 <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Deep-Learning-Techniques-and-Vision-Transform/assets/90696437/a380ab8f-bcef-4fa0-996c-9b3f60f037c9" alt="Single defect"  height =900 width="700"/>
        <div align="center"> <figcaption>Single defect detection model architecture </figcaption> </div>
        </figure>
</div>          
                           
## The model results:
 <div align="center">
 <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Deep-Learning-Techniques-and-Vision-Transform/assets/90696437/666072bb-e768-4371-88c0-6c4b82d08ae2" alt="Single defect"  height =400 width="500"/>
        <div align="center"> <figcaption>The accuracy and the validation metrics </figcaption> </div>
        </figure>
</div>   
                           
                           
# Single detection of a defect using custom CNN model
                           
## The model architecture:
 <div align="center">
 <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Deep-Learning-Techniques-and-Vision-Transform/assets/90696437/9595c6d2-f88e-495f-94c3-c4febf16008e" alt="Single defect"  height =1500 width="600"/>
        <div align="center"> <figcaption>Multiple defect detection model architecture </figcaption> </div>
        </figure>
</div>  
                           
## The model results:
 <div align="center">
 <figure>
        <img src="https://github.com/toqaalaa20/Automated-Detection-of-Defects-on-Metal-Surfaces-using-Deep-Learning-Techniques-and-Vision-Transform/assets/90696437/b278ed0d-2186-4992-b304-8a55985e34ea" alt="Single defect"  height =400 width="500"/>
        <div align="center"> <figcaption>The accuracy and the validation metrics </figcaption> </div>
        </figure>
</div>          




