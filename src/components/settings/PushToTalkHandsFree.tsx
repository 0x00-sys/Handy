import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface PushToTalkHandsFreeProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const PushToTalkHandsFree: React.FC<PushToTalkHandsFreeProps> =
  React.memo(({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const handsFreeEnabled = getSetting("push_to_talk_hands_free") || false;

    return (
      <ToggleSwitch
        checked={handsFreeEnabled}
        onChange={(enabled) =>
          updateSetting("push_to_talk_hands_free", enabled)
        }
        isUpdating={isUpdating("push_to_talk_hands_free")}
        label={t("settings.general.pushToTalkHandsFree.label")}
        description={t("settings.general.pushToTalkHandsFree.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  });
