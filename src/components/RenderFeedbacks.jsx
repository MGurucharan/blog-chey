import AnimatedContent from "./AnimatedContent";
import SpotlightCard from "./SpotlightCard";

const RenderFeedbacks = (POCfeedbacks) => {
  return (
    <div className="space-y-5">
      {POCfeedbacks.value.map((feedback,index) => (
        <AnimatedContent
          key={index}
          distance={30}
          direction="vertical"
          duration={0.6}
          delay={0.3}
          threshold={0.1}
        >
          <SpotlightCard
            className="text-white"
            spotlightColor="rgba(255, 200, 0, 0.2)"
          >
            <div className="flex flex-col">
              <div className="flex items-center mb-1">
                <div className="w-1 h-1 sm:w-2 sm:h-2 rounded-full bg-gradient-to-r from-orange-500 to-yellow-500 mr-3"></div>
                <div>
                  <p className="text-[15px]">{feedback.name}</p>
                  <p className="text-xs text-neutral-400">3 days ago</p>
                </div>
              </div>
              <p className="text-neutral-300 text-sm sm:text-base">
                {feedback.desp}
              </p>
            </div>
          </SpotlightCard>
        </AnimatedContent>
      ))}
    </div>
  );
};

export default RenderFeedbacks;
