
## Decision trees model

以 cat classification 为例（判断一个东西是不是猫）：

![|650](files/DecisionTreesCatExample.png)

特征只取几个离散值（如耳朵形状只有pointy和floppy两类）

从表中可以绘制这个例子的**决策树**(Decision Tree)如下

```mermaid
graph TD

A([Ear shape])
B([Face shape])
C([Whiskers])
D(Cat)
E(Not cat)
F(Cat)
G(Not cat)

A -->|Pointy| B
A -->|Floppy| C
B -->|Round| D
B -->|Not round| E
C -->|Present| F
C -->|Absent| G
```

*上面的所有的椭圆节点称为**决策节点**(Decision nodes)*

## Learning Process

*the overall process of what you need to do to build a decision tree*

1. Decide what feature to use at the root node

For example, the "Ear Shape" in the Cat Classification. 
We split the training examples according to the value of the ear shape feature.

2. Focus just on the left part / left branch of the decision tree to decide what nodes to put over there. In particular, what feature that we want to split on or what feature do we want to use next.

for example, the "Face shape" for the left branch

3. repeat a similar process on the right part / right branch

for example, the "Whiskers" for the right branch


- Decision 1: How to choose what feature to split on at each node?

Maximize purity (or minimize impurity) 即尽量朝着左子树全部是cat而右子树没有cat的目标迈进

- Decision 2: When do you stop splitting?

When a node is 100% one class; 

When splitting a node will result in the tree exceeding a maximum depth (this maximum depth is a parameter that you could say) 
*The reason why we limit the depth of the decision tree is to make sure for us the tree doesn't get too big and unwieldy and make sure the tree is less prone to overfitting.*; 

When improvements in purity score are below a threshold; 

When number of examples in a node is below a threshold.

