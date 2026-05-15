class RewardFunction:
    LANE_WIDTH = 2.0
    MAX_HEADING_ERR = 1.0

    def compute(self, info):
        reward = 0
        
        if info.get("out_of_road", False):
            return -300.0
        
        if info.get("crash", False):
            return -300.0
        # 1. Add a small 'Existence' bonus. 
        # This makes every step alive worth something.
        reward += 0.1

        reward += self.forward_reward(info)
        reward += self.idle_penalty(info) # Renamed for clarity
        reward += self.lane_reward(info)
        reward += self.goal_reward(info)
        reward += self.action_smoothing_penalty(info)
        reward += self.heading_penalty(info)

        return reward
    
    def idle_penalty(self, info):
        # Reduced this so it doesn't completely cancel out the lane reward
        if info.get("speed", 0.00) < 0.5:
            return -0.5 
        return 0.0
    
    def forward_reward(self, info):
        # Boosted slightly to reward progress more than just "sitting centered"
        return info.get("speed", 0.00) * 0.03
    
    def lane_reward(self, info):
        lane_offset = abs(info.get("lateral", 0.0))
        normalised = min(lane_offset / self.LANE_WIDTH, 1.0)
        # We give up to +0.8 here
        return (1.0 - (normalised)) * 1.5
    
    def action_smoothing_penalty(self, info):
        # This was too high. -1.0 * steering is a huge penalty.
        # Lowered to -0.1 so it discourages 'jitter' without stopping 'turning'.
        return float(-0.1 * abs(info.get("steering", 0.0)))
    
    def heading_penalty(self, info):
        heading_err = abs(info.get("heading_diff", 0.0))
        normilised = min(heading_err / self.MAX_HEADING_ERR, 1.0)

        return -0.5 * normilised


    def goal_reward(self, info):

        if info.get("arrive_dest", False):
            return 150.0
        
        return 0.0
