"""校园社交网络专用建模求解器。

针对“好友关系与社交价值”类社交网络题目，提供四个关键模型：
- 社群发现与高密度社群挖掘（问题一）
- 基于链路预测的好友推荐（问题二）
- 信息传播（独立级联）仿真与关键用户筛选（问题三）
- 推送名额优化的影响力最大化（问题四）
"""

from models.social_network.campus import run_campus_social_analysis

__all__ = ["run_campus_social_analysis"]
