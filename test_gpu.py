    def step(self, action):
        # Capture the initial state of Poké Balls before executing the action
        initial_pokeballs = self.find_pokeballs_quantities()

        # Execute the action
        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()
        
        # After the action has been executed and you want to check for changes
        final_pokeballs = self.find_pokeballs_quantities()

        # Calculate rewards based on the changes in Poké Ball quantities
        pokeball_reward_points = 0
        for pokeball_id in initial_pokeballs:
            gained = final_pokeballs[pokeball_id] - initial_pokeballs[pokeball_id]
            if gained > 0:  # Only reward for net gains
            points = {0x01: 4, 0x02: 3, 0x03: 2, 0x04: 1}.get(pokeball_id, 0)
            pokeball_reward_points += gained * points

        # Adjust your reward based on pokeball_reward_points
        new_reward += pokeball_reward_points * 0.1  # Example adjustment


        # Depending on exploration strategy, update the frame index or seen coordinates
        if self.use_screen_explore:
            # trim off memory from frame for knn index
            frame_start = 2 * (self.memory_height + self.mem_padding)
            obs_flat = obs_memory[frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)
            self.update_frame_knn_index(obs_flat)
        else:
            self.update_seen_coords()
            
        # Update heal rewards, party size, and calculate new rewards based on game state changes
        self.update_heal_reward()
        self.party_size = self.read_m(PARTY_SIZE_ADDRESS)
        new_reward, new_prog = self.update_reward()
    
        # Update health status for future decisions
        self.last_health = self.read_hp_fraction()

        # Update short term reward memory based on progress
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        # Determine if step limit or other end condition has been reached
        step_limit_reached = self.check_if_done()

        # Logging and information saving
        self.save_and_print_info(step_limit_reached, obs_memory)

        # Increment step count for tracking
        self.step_count += 1

        # Monitor item bag, stored items, and final state of Poké Balls after the action
        item_bag_contents = self.monitor_item_bag()
        stored_items_contents = self.monitor_stored_items()
        final_pokeballs = self.find_pokeballs_quantities()

        # Prepare additional info with the most current state for decision making or logging
        info = {
            "item_bag_contents": item_bag_contents,
            "stored_items_contents": stored_items_contents,
            "pokeballs_quantities": final_pokeballs  # Use final quantities for the most current state
        }

        #Return observation, scaled reward, completion flag, and info
        return obs_memory, new_reward, False, step_limit_reached, info

    def find_pokeballs_quantities(self, initial_quantities=None):
        item_start_address = 0xD31E  # Item IDs start at $D31E, right after the total number of items byte
        pokeball_ids = [0x01, 0x02, 0x03, 0x04]  # IDs for Master Ball, Ultra Ball, Great Ball, Poké Ball
        quantities = {id: 0 for id in pokeball_ids}  # Initialize quantities for each type of ball

        for i in range(20):  # 20 is the maximum number of item slots in the bag
            item_id_address = item_start_address + i * 2
            item_id = self.read_m(item_id_address)
            if item_id in pokeball_ids:
                quantity_address = item_id_address + 1
                quantity = self.read_m(quantity_address)
                quantities[item_id] += quantity
            elif item_id == 0xFF:
                break
        
        return quantities
