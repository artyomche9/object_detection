class AddIdentifiedObjectToImage < ActiveRecord::Migration[5.2]
  def change
    add_column :images, :identified_object, :string
  end
end
