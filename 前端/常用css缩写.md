## 常用css属性缩写
### 对于和盒子四边相关的模型（margin和padding等）遵循时钟模型，即顺时针，上->左->右->下
- 4个值：上 左 右 下
- 3个值：上 左右 下
- 2个值：上下 左右
- 1个值：上左右下
### 其他
#### #border边框
3个属性：border-width，border-style,border-color
说明：三个属性可以随意调整位置
示例：
~~~css
/*边框宽度 边框类型 边框颜色*/
border: 1px soild black;
~~~

#### #border-radius边框圆角
4个属性：border-top-left-radius,border-top-right-radius,border-bottom-right-radius,border-bottom-left-radius
说明：类似于时钟模型，但是方向是左上->右上->右下->左下
示例1
~~~css
/*表示所有边框的的角的边长都是10*/
border-radius： 10px;
/*表示左上和右下是10px*/
border-radius： 10px 20px;
/**/
border-radius： 10px 20px 30px;
/**/
border-radius： 10px 20px 30px 40px;
~~~
注意：这里的2个值和上面的不一样，这个是对角线是一对的，而上面是左右是一对；
示例2：
~~~css
/* /表示是4个属性的前半和后半 */
border-radius: 4px 3px 6px / 2px 4px;
/*
表示：
border-top-left-radius: 4px 2px; 
border-top-right-radius: 3px 4px; 对角线
border-bottom-right-radius: 6px 2px; 
border-bottom-left-radius: 3px 4px; 对角线
*/
~~~

#### #background背景
8个属性：background-color,nackground-image,background-repeat,background-position,background-attchment,background-size,background-origin,background-clip
说明：值的顺序比较灵活，但又遵循一定的习惯，background-size一定要跟在background-position后面，用/分隔
示例：
~~~css
/*包含照片和颜色的f'z*/



~~~