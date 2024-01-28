

for first_image, second_image, label, unormalized_label in train_loader:
                first_image, second_image, label, unormalized_label = first_image.to(
                    device), second_image.to(device), label.to(device), unormalized_label.to(device)

                # Forward pass
                unnormalized_output, output, penalty = self.forward(
                    first_image, second_image)
                # Compute loss
                l2_loss = self.L2_loss(output, label)
                loss = l2_loss + penalty 
                avg_loss += loss.detach()

                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



def forward(self, x1, x2):
    x1 = self.clip_image_processor(images=x1,   return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)

    x2 = self.clip_image_processor(images=x2, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)

    x1_embeddings = self.pretrained_model(**x1).last_hidden_state[:, :49, :].view(-1, 7*7*768).to(device)
    x2_embeddings = self.pretrained_model(**x2).last_hidden_state[:, :49, :].view(-1, 7*7*768).to(device)


    # Concatenate both original and rotated embedding vectors embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1).to(device)
    embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1).to(device)
    # Train MLP on embedding vectors            
    output = self.mlp(embeddings).to(device)
    return output.view(-1,3,3), output.view(-1,3,3), 0


class CustomDataset(torch.utils.data.Dataset):
    def _init_(self, sequence_path, poses, transform, K):
        self.sequence_path = sequence_path
        self.sequence_num = sequence_path.split('/')[1]
        self.poses = poses
        self.transform = transform
        self.k = K
        self.valid_indices = self.get_valid_indices()

    def _len_(self):
        return len(self.valid_indices) - jump_frames

    def get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.poses) - jump_frames):
            img1_path = os.path.join(self.sequence_path, f'{idx:06}.png')
            img2_path = os.path.join(self.sequence_path, f'{idx+jump_frames:06}.png')
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                valid_indices.append(idx)
        return valid_indices

    def _getitem_(self, idx):
        # If one of the frames is "Bad"- skip
        idx = self.valid_indices[idx]
        img1_path = os.path.join(self.sequence_path, f'{idx:06}.png')
        img2_path = os.path.join(self.sequence_path, f'{idx+jump_frames:06}.png')

        # Create PIL images
        original_first_image = Image.open(img1_path)
        original_second_image = Image.open(img2_path)

        # Transform: Resize, center, grayscale
        first_image = self.transform(original_first_image).to(device)
        second_image = self.transform(original_second_image).to(device)

        return first_image, second_image, torch.rand(3,3).to(device), torch.rand(3,3).to(device)